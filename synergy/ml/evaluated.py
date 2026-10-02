import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings

logger = logging.getLogger(__name__)
FRESH_DAYS = 14
FRESH_SECONDS = FRESH_DAYS * 86400


class EvaluatedPlayers:
    def __init__(self, table, client, bucket: str, folder: Path | None = None, clock=time.time):
        self.table, self.client, self.bucket, self.clock = table, client, bucket, clock
        self.folder = folder or Path(tempfile.gettempdir()) / "lolpredictor-evaluated"
        self.held: dict[str, tuple[int, dict, dict]] = {}

    def puuid_of(self, riot_id: str) -> str | None:
        from ..api.evaluate import NAMES, name_key

        found = self.table.get(NAMES, name_key(riot_id))
        return found["puuid"] if found else None

    def state(self, puuid: str) -> dict | None:
        from ..api.evaluate import EVAL, READY

        found = self.table.get(EVAL, puuid)
        if found and found.get("status") == READY and int(found.get("expires", 0)) > int(self.clock()):
            return found
        return None

    def find(self, riot_id: str) -> tuple[dict, dict] | None:
        puuid = self.puuid_of(riot_id)
        if puuid is None:
            return None
        return self.load(puuid)

    def load(self, puuid: str) -> tuple[dict, dict] | None:
        state = self.state(puuid)
        if state is None:
            return None
        stamp = int(state.get("finished", 0) or 0)
        held = self.held.get(puuid)
        if held and held[0] == stamp:
            return held[1], held[2]
        self.folder.mkdir(parents=True, exist_ok=True)
        try:
            profile = json.loads(self.client.get_object(Bucket=self.bucket, Key=f"profiles/{puuid}.json")["Body"].read())
            target = self.folder / f"{puuid}.npz"
            self.client.download_file(self.bucket, f"vectors/{puuid}.npz", str(target))
            with np.load(target, allow_pickle=True) as data:
                vectors = {
                    "columns": [str(name) for name in data["columns"]],
                    "position": [str(name) for name in data["position"]],
                    "seats": data["seats"].astype(int),
                    "evidence": data["evidence"].astype(float),
                    "matrix": data["matrix"].astype(np.float32),
                }
        except Exception:
            logger.exception("could not load the evaluated player %s", puuid[:8])
            return None
        profile["evaluated_at"] = stamp
        self.held[puuid] = (stamp, profile, vectors)
        return profile, vectors


def profile_row(profile: dict) -> pd.Series:
    row = {key: value for key, value in profile.items() if key not in ("positions", "seats", "champions")}
    row.setdefault("games", 0)
    row.setdefault("winrate", 0.0)
    row["evaluated"] = True
    return pd.Series(row, name=profile["puuid"])


def seats_of(vectors: dict, columns: list[str]) -> dict[tuple[str, str], tuple[np.ndarray, int, float]]:
    order = pd.Index(vectors["columns"]).get_indexer(columns)
    if (order < 0).any():
        raise ValueError("the evaluated vectors do not match the served columns")
    puuid = vectors.get("puuid")
    return {
        (puuid, position): (vectors["matrix"][index][order], int(vectors["seats"][index]), float(vectors["evidence"][index]))
        for index, position in enumerate(vectors["position"])
    }


def pool_effect(settings: Settings, position: str, champions: dict[str, int]) -> float:
    from .serving import champion_effects

    effects = champion_effects(settings)
    if effects is None or not champions:
        return 0.0
    total, weighted = 0, 0.0
    for champion, games in champions.items():
        effect = effects.get((position, champion))
        if effect is not None:
            weighted += effect * games
            total += games
    return weighted / total if total else 0.0
