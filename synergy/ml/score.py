import json
import logging
import threading
from itertools import combinations

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..features.positions import parse_position, spoken
from ..features.timeline import PAIR_TIMELINE_COLUMNS
from ..riot.routing import split_riot_id
from ..features.propensity import PROPENSITY_COLUMNS
from .dataset import PAIR_HISTORY_SOURCE, HISTORY_COLUMNS, STYLE_NAMES, phi_from_styles
from .model import SynergyModel
from .evaluated import profile_row
from .serving import adopt_player, custom_residuals, duo_between, hinge_between, known_names, lineup_between, position_profile

logger = logging.getLogger(__name__)

STYLE_COLUMNS = [f"style_{name}" for name in STYLE_NAMES]
STYLE_SUFFIXES = ("_pct", "_var")
UNINFORMATIVE = (
    "the pair model found no usable synergy signal in this corpus, so no score is reported"
)
NOT_FITTED = "The model is not ready yet. Try again later."
NO_POSITIONS = "You both play the same position. Pick a different position for one of you."
THIN = (
    "{name}'s {position} playstyle is {own}% their own evidence from {games} {noun} and {rest}% the"
    " {position} corpus row, so this score says little about them"
)


def thin_warning(name: str, position: str, games: int, evidence: float | None) -> str | None:
    if evidence is None or evidence >= 0.5:
        return None
    own = int(round(100.0 * evidence))
    return THIN.format(
        name=name, position=position, own=own, rest=100 - own, games=games, noun="game" if games == 1 else "games"
    )


UNKNOWN_NOTE = "Not in our data yet."
CANNOT_SCORE = "Can't score this friend right now."


class UnknownPlayer(LookupError):
    pass


class PairingError(ValueError):
    pass


def riot_id(profile) -> str:
    name = profile.get("game_name")
    tag = profile.get("tag_line")
    if name and tag and not pd.isna(name) and not pd.isna(tag):
        return f"{name}#{tag}"
    return f"unknown-{str(profile['puuid'])[:8]}"


class SynergyService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model: SynergyModel | None = None
        self.profiles: pd.DataFrame | None = None
        self.history: pd.DataFrame | None = None
        self.report: dict = {}
        self.propensity_report: dict = {}
        self.style_columns: list[str] = list(STYLE_COLUMNS)
        self.evaluated = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.profiles is not None

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "informative": bool(self.model is not None and self.model.informative),
            "players": 0 if self.profiles is None else int(len(self.profiles)),
            "known_pairs": 0 if self.history is None else int(len(self.history)),
            "model": self.report,
        }

    def load(self) -> "SynergyService":
        model_path = self.settings.served_model_dir / "synergy_model.pkl"
        profile_path = self.settings.served_processed_dir / "player_profiles.parquet"
        if not model_path.exists() or not profile_path.exists():
            logger.warning("model or profiles missing, service not ready")
            return self
        self.model = SynergyModel.load(model_path)
        self.profiles = pd.read_parquet(profile_path).set_index("puuid", drop=False)
        self.style_columns = [
            column
            for column in self.profiles.columns
            if column.startswith("style_")
            and not column.endswith(STYLE_SUFFIXES)
            and column != "style_confidence"
        ] or list(STYLE_COLUMNS)
        history_path = self.settings.served_processed_dir / "pair_history.parquet"
        if history_path.exists():
            self.history = pd.read_parquet(history_path).set_index("pair_key", drop=False)
        propensity_path = self.settings.served_processed_dir / "propensity_report.json"
        if propensity_path.exists():
            with open(propensity_path, encoding="utf-8") as handle:
                self.propensity_report = json.load(handle)
        report_path = self.settings.served_model_dir / "training_report.json"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as handle:
                self.report = json.load(handle)
        return self

    def _require(self) -> tuple[SynergyModel, pd.DataFrame]:
        if self.model is None or self.profiles is None:
            raise RuntimeError("synergy model is not trained yet")
        return self.model, self.profiles

    def resolve(self, query: str) -> pd.Series:
        _, profiles = self._require()
        query = query.strip()
        if query in profiles.index:
            return profiles.loc[query]
        if "#" in query:
            name, tag = split_riot_id(query)
            found = profiles[
                (profiles["game_name"].str.lower() == name.lower())
                & (profiles["tag_line"].str.lower() == tag.lower())
            ]
        else:
            found = profiles[profiles["game_name"].str.lower() == query.lower()]
        if found.empty:
            try:
                return self._resolve_alias(query)
            except UnknownPlayer:
                adopted = self.adopt(query)
                if adopted is None:
                    raise
                return adopted
        return found.iloc[0]

    def adopt(self, query: str) -> pd.Series | None:
        if self.evaluated is None or self.profiles is None:
            return None
        found = self.evaluated.find(query)
        return None if found is None else self._adopt(*found)

    def adopt_puuid(self, puuid: str) -> pd.Series | None:
        if self.evaluated is None or self.profiles is None:
            return None
        found = self.evaluated.load(puuid)
        return None if found is None else self._adopt(*found)

    def _adopt(self, profile: dict, vectors: dict) -> pd.Series:
        row = profile_row(profile)
        if row.name not in self.profiles.index:
            self.profiles = pd.concat([self.profiles, self._as_profile(row)])
            self.profiles["puuid"] = self.profiles.index
        adopt_player(self.settings, profile, vectors)
        return self.profiles.loc[row.name]

    def _as_profile(self, row: pd.Series) -> pd.DataFrame:
        frame = row.to_frame().T.reindex(columns=self.profiles.columns.union(row.index, sort=False))
        for column, dtype in self.profiles.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                blank = 50.0 if column.endswith("_pct") else 0.0
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(blank).astype(dtype)
        return frame

    def _resolve_alias(self, query: str) -> pd.Series:
        _, profiles = self._require()
        if "#" not in query:
            raise UnknownPlayer(query)
        name, tag = split_riot_id(query)
        names = known_names(self.settings)
        puuid = names.get((name.lower(), tag.lower())) if names is not None else self._alias_in_database(name, tag)
        if puuid is None or puuid not in profiles.index:
            raise UnknownPlayer(query)
        return profiles.loc[puuid]

    def _alias_in_database(self, name: str, tag: str) -> str | None:
        from ..ingest.store import Store

        store = Store(self.settings)
        try:
            player = store.find_player_by_riot_id(name, tag)
        finally:
            store.close()
        return player["puuid"] if player else None

    def search(self, term: str = "", limit: int = 25) -> list[dict]:
        _, profiles = self._require()
        frame = profiles[profiles["game_name"].notna()]
        if term:
            frame = frame[frame["game_name"].str.contains(term, case=False, na=False, regex=False)]
        frame = frame.sort_values("games", ascending=False).head(limit)
        return [
            {
                "puuid": row["puuid"],
                "riot_id": riot_id(row),
                "tier": row.get("tier"),
                "games": int(row["games"]),
                "winrate": round(float(row["winrate"]), 4),
                "main_position": row.get("main_position"),
            }
            for _, row in frame.iterrows()
        ]

    def _pair_history(self, left: str, right: str) -> dict:
        if self.history is None:
            return {}
        key = "|".join(sorted([left, right]))
        if key not in self.history.index:
            return {}
        return self.history.loc[key].to_dict()

    def _history_rows(self, anchor: str, others: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(0.0, index=range(len(others)), columns=HISTORY_COLUMNS)
        if self.history is None:
            return frame
        keys = ["|".join(sorted([anchor, other])) for other in others]
        known = self.history.reindex(keys)
        games = np.nan_to_num(known["games"].to_numpy(dtype=float))
        present = games > 0
        frame["hist_present"] = present.astype(float)
        frame["hist_games_log"] = np.log1p(games)
        for column in PAIR_HISTORY_SOURCE:
            values = (
                np.nan_to_num(known[column].to_numpy(dtype=float))
                if column in known.columns
                else np.zeros(len(others))
            )
            frame[f"hist_{column}"] = np.where(present, values, 0.0)
        return frame

    def build_phi(self, anchor: pd.Series, others: pd.DataFrame) -> pd.DataFrame:
        columns = self.style_columns
        left = np.tile(anchor[columns].to_numpy(dtype=float), (len(others), 1))
        right = others[columns].to_numpy(dtype=float)
        history = self._history_rows(anchor["puuid"], others["puuid"].tolist())
        return phi_from_styles(left, right, history)

    def _positions(self, a: pd.Series, b: pd.Series, left: str | None, right: str | None, required: bool) -> dict | None:
        chosen = {
            "left": parse_position(left) if left else str(a.get("main_position") or ""),
            "right": parse_position(right) if right else str(b.get("main_position") or ""),
        }
        if chosen["left"] and chosen["right"] and chosen["left"] != chosen["right"]:
            return chosen
        if not required:
            return None
        if not chosen["left"] or not chosen["right"]:
            raise PairingError(f"We don't know which position {riot_id(a) if not chosen['left'] else riot_id(b)} plays. Pick one.")
        raise PairingError(f"{riot_id(a)} and {riot_id(b)} both play {spoken(chosen['left'])}. Pick a different position for one of you.")

    def _interaction(self, a: pd.Series, b: pd.Series, positions: dict | None, required: bool = True, customs=None) -> dict | None:
        if positions is None:
            return None
        for profile, side in ((a, "left"), (b, "right")):
            known = position_profile(profile["puuid"], positions[side], self.settings)
            if known is None:
                return None
            if known["games"] == 0:
                if not required:
                    return None
                raise PairingError(f"{riot_id(profile)} has no games as {spoken(positions[side])} in our data.")
        games = customs(a["puuid"], b["puuid"]) if customs else []
        return duo_between(a["puuid"], positions["left"], b["puuid"], positions["right"], self.settings, custom_residuals(games, self.settings))

    @staticmethod
    def _warnings(a: pd.Series, b: pd.Series, interaction: dict | None) -> list[str]:
        if not interaction:
            return []
        found = []
        for profile, side in ((a, "left"), (b, "right")):
            warning = thin_warning(
                riot_id(profile), interaction["positions"][side], interaction[f"{side}_games"], interaction.get(f"{side}_evidence")
            )
            if warning:
                found.append(warning)
        return found

    def pair_score(
        self,
        left: str,
        right: str,
        left_position: str | None = None,
        right_position: str | None = None,
        positions_required: bool = True,
        details: bool = True,
        customs=None,
    ) -> dict:
        self._require()
        a = self.resolve(left)
        b = self.resolve(right)
        if a["puuid"] == b["puuid"]:
            raise PairingError("a player cannot be paired with themselves")
        positions = self._positions(a, b, left_position, right_position, positions_required)
        history = self._pair_history(a["puuid"], b["puuid"]) if details else {}
        games = int(history.get("games", 0) or 0)
        wins = float(history.get("wins", 0) or 0)
        interaction = self._interaction(a, b, positions, positions_required, customs)
        return {
            "score": interaction["score"] if interaction else None,
            "projected_gold": interaction["projected_gold"] if interaction else None,
            "minute": interaction["minute"] if interaction else None,
            "reliable": bool(interaction and interaction["reliable"]),
            "note": interaction["note"] if interaction else (NOT_FITTED if positions else NO_POSITIONS),
            "warnings": self._warnings(a, b, interaction),
            "games_together": games if details else None,
            "winrate_together": round(wins / games, 4) if games else None,
            "positions": positions,
            "hinge": hinge_between(a["puuid"], b["puuid"], self.settings) if details else None,
            "interaction": interaction,
            "players": [self.player_summary(a), self.player_summary(b)],
            "shared_play": {
                key: round(float(history[key]), 4)
                for key in PAIR_TIMELINE_COLUMNS
                if key in history and not pd.isna(history[key])
            },
        }

    def player_summary(self, profile: pd.Series) -> dict:
        return {
            "puuid": profile["puuid"],
            "riot_id": riot_id(profile),
            "tier": profile.get("tier"),
            "division": profile.get("division"),
            "games": int(profile["games"]),
            "winrate": round(float(profile["winrate"]), 4),
            "main_position": profile.get("main_position"),
            "main_champion": profile.get("main_champion"),
            "champion_pool": int(profile.get("champion_pool", 0)),
            "style_confidence": round(float(profile.get("style_confidence", 1.0)), 3),
            "style": {
                name: round(float(profile.get(f"style_{name}_pct", 50.0)), 1) for name in STYLE_NAMES
            },
            "tendencies": {
                name: self._tendency(profile, name)
                for name in (column.removeprefix("prop_") for column in PROPENSITY_COLUMNS)
            },
        }

    def _tendency(self, profile: pd.Series, name: str) -> dict:
        measurable = float(self.propensity_report.get(name, {}).get("player_dispersion", 0.0)) > 0.01
        entry = {
            "measurable": measurable,
            "chances": int(profile.get(f"chances_{name}", 0) or 0),
        }
        if measurable:
            entry["percentile"] = round(float(profile.get(f"prop_{name}_pct", 50.0)), 1)
            entry["log_ratio"] = round(float(profile.get(f"prop_{name}", 0.0) or 0.0), 3)
        else:
            entry["note"] = "not separable from chance in this corpus"
        return entry

    def player_report(self, query: str) -> dict:
        profile = self.resolve(query)
        summary = self.player_summary(profile)
        return summary

    def team_report(self, queries: list[str]) -> dict:
        profiles = [self.resolve(query) for query in queries]
        if len(profiles) < 2:
            raise PairingError("at least two players are required")
        entries = []
        for left, right in combinations(range(len(profiles)), 2):
            result = self.pair_score(profiles[left]["puuid"], profiles[right]["puuid"], positions_required=False)
            interaction = result["interaction"]
            entries.append(
                {
                    "a": profiles[left]["puuid"],
                    "b": profiles[right]["puuid"],
                    "a_riot_id": riot_id(profiles[left]),
                    "b_riot_id": riot_id(profiles[right]),
                    "positions": result["positions"],
                    "score": result["score"],
                    "projected_gold": result["projected_gold"],
                    "percentile": interaction["percentile"] if interaction else None,
                    "reliable": result["reliable"],
                    "games_together": result["games_together"],
                }
            )
        scored = [entry for entry in entries if entry["score"] is not None]
        complete = bool(scored) and len(scored) == len(entries)
        reliable = complete and all(entry["reliable"] for entry in scored)
        return {
            "team_score": round(float(np.mean([entry["score"] for entry in scored])), 1) if complete else None,
            "reliable": reliable,
            "strongest": max(scored, key=lambda item: item["score"]) if complete else None,
            "weakest": min(scored, key=lambda item: item["score"]) if complete else None,
            "pairs": entries,
            "players": [self.player_summary(profile) for profile in profiles],
        }

    def lineup(self, players: dict[str, str]) -> dict:
        self._require()
        assignments = {parse_position(position): self.resolve(query) for position, query in players.items()}
        if len(assignments) != len(players):
            raise PairingError("each position may be given once")
        if len({profile["puuid"] for profile in assignments.values()}) != len(assignments):
            raise PairingError("a player cannot fill two positions")
        names = {profile["puuid"]: riot_id(profile) for profile in assignments.values()}
        summaries = {position: self.player_summary(profile) for position, profile in assignments.items()}
        warnings = []
        for position, profile in assignments.items():
            known = position_profile(profile["puuid"], position, self.settings)
            if known is None:
                return {"reliable": False, "note": NOT_FITTED, "players": summaries}
            if known["games"] == 0:
                raise PairingError(f"{riot_id(profile)} has no games as {spoken(position)} in our data.")
            warning = thin_warning(riot_id(profile), position, known["games"], known["evidence"])
            if warning:
                warnings.append(warning)
        result = lineup_between({position: profile["puuid"] for position, profile in assignments.items()}, self.settings)
        if result is None:
            return {"reliable": False, "note": NOT_FITTED, "players": summaries}
        for pair in result["pairs"]:
            pair["left"], pair["right"] = names[pair["left"]], names[pair["right"]]
        return {**result, "warnings": warnings, "players": summaries}

    def friends(self, me: str, friends: list[str], me_position: str | None = None, details: bool = True, customs=None) -> dict:
        self._require()
        anchor = self.resolve(me)
        rows, own = [], None
        for entry in friends:
            name, _, wanted = entry.strip().partition(":")
            name = name.strip()
            if not name:
                continue
            try:
                found = self.pair_score(me, name, me_position, wanted.strip() or None, details=details, customs=customs)
            except UnknownPlayer:
                rows.append({"riot_id": name, "note": UNKNOWN_NOTE})
                continue
            except PairingError as error:
                rows.append({"riot_id": name, "note": str(error)})
                continue
            except Exception:
                logger.exception("could not score %s with %s", me, name)
                rows.append({"riot_id": name, "note": CANNOT_SCORE})
                continue
            friend, interaction = found["players"][1], found["interaction"]
            if interaction is None:
                rows.append({"riot_id": friend["riot_id"], "note": found["note"]})
                continue
            if own is None:
                own = {"position": interaction["positions"]["left"], "reading": interaction["edge"]["left"], "evidence": interaction["left_evidence"], "games": interaction["left_games"], "minute": interaction["minute"]}
            rows.append(
                {
                    "riot_id": friend["riot_id"],
                    "position": interaction["positions"]["right"],
                    "score": interaction["score"],
                    "projected_gold": interaction["projected_gold"],
                    "record": interaction["edge"].get("record"),
                    "reading": interaction["edge"]["right"],
                    "fit": interaction["edge"]["fit"],
                    "total": interaction["edge"]["total"],
                    "games": interaction["right_games"],
                    "evidence": interaction["right_evidence"],
                    "games_together": found["games_together"],
                    "standout": next(iter(interaction["reading"]["right"]["distinctive"]), None),
                    "thin": any(friend["riot_id"] in warning for warning in found["warnings"]),
                    "note": None,
                }
            )
        ranked = sorted(rows, key=lambda row: (-row["score"], -row["total"]) if row.get("score") is not None else (float("inf"), 0.0))
        return {"me": {"riot_id": riot_id(anchor), **(own or {})}, "friends": ranked}

    def best_partners(self, query: str, limit: int = 10) -> dict:
        model, profiles = self._require()
        target = self.resolve(query)
        if not model.informative:
            return {
                "player": self.player_summary(target),
                "reliable": False,
                "note": UNINFORMATIVE,
                "best": [],
                "worst": [],
            }
        candidates = profiles[
            (profiles["puuid"] != target["puuid"]) & profiles["game_name"].notna()
        ].reset_index(drop=True)
        phi = self.build_phi(target, candidates)
        scores = model.score(model.synergy(phi))
        frame = pd.DataFrame(
            {
                "puuid": candidates["puuid"],
                "riot_id": [riot_id(row) for _, row in candidates.iterrows()],
                "main_position": candidates["main_position"],
                "games": candidates["games"].astype(int),
                "score": np.round(scores, 1),
            }
        ).sort_values("score", ascending=False)
        return {
            "player": self.player_summary(target),
            "reliable": True,
            "best": frame.head(limit).to_dict(orient="records"),
            "worst": frame.tail(limit).sort_values("score").to_dict(orient="records"),
        }


_service: SynergyService | None = None
_service_lock = threading.RLock()


def get_service() -> SynergyService:
    global _service
    with _service_lock:
        if _service is None or not _service.ready:
            _service = SynergyService().load()
        return _service


def reload_service() -> SynergyService:
    global _service
    with _service_lock:
        _service = SynergyService().load()
        return _service
