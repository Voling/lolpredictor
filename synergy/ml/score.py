import json
import logging
from itertools import combinations

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..features.positions import parse_position
from ..features.timeline import PAIR_TIMELINE_COLUMNS
from ..riot.routing import split_riot_id
from ..features.propensity import PROPENSITY_COLUMNS
from .dataset import PAIR_HISTORY_SOURCE, HISTORY_COLUMNS, STYLE_NAMES, phi_from_styles
from .model import SynergyModel
from .serving import hinge_between, known_names, lineup_between, pair_between, position_profile

logger = logging.getLogger(__name__)

STYLE_COLUMNS = [f"style_{name}" for name in STYLE_NAMES]
STYLE_SUFFIXES = ("_pct", "_var")
UNINFORMATIVE = (
    "the pair model found no usable synergy signal in this corpus, so no score is reported"
)
NOT_FITTED = "the pair matrix is not fitted yet"
NO_POSITIONS = "the two players share a main position, so no position pair was read"
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


class UnknownPlayer(LookupError):
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
        model_path = self.settings.model_dir / "synergy_model.pkl"
        profile_path = self.settings.processed_dir / "player_profiles.parquet"
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
        history_path = self.settings.processed_dir / "pair_history.parquet"
        if history_path.exists():
            self.history = pd.read_parquet(history_path).set_index("pair_key", drop=False)
        propensity_path = self.settings.processed_dir / "propensity_report.json"
        if propensity_path.exists():
            with open(propensity_path, encoding="utf-8") as handle:
                self.propensity_report = json.load(handle)
        report_path = self.settings.model_dir / "training_report.json"
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
            return self._resolve_alias(query)
        return found.iloc[0]

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
            frame = frame[frame["game_name"].str.contains(term, case=False, na=False)]
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
        raise ValueError(
            f"{riot_id(a)} and {riot_id(b)} are both given {chosen['left']}, a duo needs two different"
            " positions, pass the position each will play"
        )

    def _interaction(self, a: pd.Series, b: pd.Series, positions: dict | None, required: bool = True) -> dict | None:
        if positions is None:
            return None
        for profile, side in ((a, "left"), (b, "right")):
            known = position_profile(profile["puuid"], positions[side], self.settings)
            if known is None:
                return None
            if known["games"] == 0:
                if not required:
                    return None
                raise ValueError(f"{riot_id(profile)} has no games as {positions[side]} in the corpus")
        return pair_between(a["puuid"], positions["left"], b["puuid"], positions["right"], self.settings)

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
    ) -> dict:
        self._require()
        a = self.resolve(left)
        b = self.resolve(right)
        if a["puuid"] == b["puuid"]:
            raise ValueError("a player cannot be paired with themselves")
        positions = self._positions(a, b, left_position, right_position, positions_required)
        history = self._pair_history(a["puuid"], b["puuid"])
        games = int(history.get("games", 0) or 0)
        wins = float(history.get("wins", 0) or 0)
        interaction = self._interaction(a, b, positions, positions_required)
        return {
            "score": interaction["score"] if interaction else None,
            "projected_gold_at_15": interaction["projected_gold_at_15"] if interaction else None,
            "reliable": bool(interaction and interaction["reliable"]),
            "note": interaction["note"] if interaction else (NOT_FITTED if positions else NO_POSITIONS),
            "warnings": self._warnings(a, b, interaction),
            "games_together": games,
            "winrate_together": round(wins / games, 4) if games else None,
            "positions": positions,
            "hinge": hinge_between(a["puuid"], b["puuid"], self.settings),
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
            raise ValueError("at least two players are required")
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
                    "projected_gold_at_15": result["projected_gold_at_15"],
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
            raise ValueError("each position may be given once")
        if len({profile["puuid"] for profile in assignments.values()}) != len(assignments):
            raise ValueError("a player cannot fill two positions")
        names = {profile["puuid"]: riot_id(profile) for profile in assignments.values()}
        summaries = {position: self.player_summary(profile) for position, profile in assignments.items()}
        warnings = []
        for position, profile in assignments.items():
            known = position_profile(profile["puuid"], position, self.settings)
            if known is None:
                return {"reliable": False, "note": NOT_FITTED, "players": summaries}
            if known["games"] == 0:
                raise ValueError(f"{riot_id(profile)} has no games as {position} in the corpus")
            warning = thin_warning(riot_id(profile), position, known["games"], known["evidence"])
            if warning:
                warnings.append(warning)
        result = lineup_between({position: profile["puuid"] for position, profile in assignments.items()}, self.settings)
        if result is None:
            return {"reliable": False, "note": NOT_FITTED, "players": summaries}
        for pair in result["pairs"]:
            pair["left"], pair["right"] = names[pair["left"]], names[pair["right"]]
        return {**result, "warnings": warnings, "players": summaries}

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


def get_service() -> SynergyService:
    global _service
    if _service is None:
        _service = SynergyService().load()
    return _service


def reload_service() -> SynergyService:
    global _service
    _service = SynergyService().load()
    return _service
