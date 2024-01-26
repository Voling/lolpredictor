import numpy as np
import pandas as pd

CHAMPION_PRIOR = 10.0
CLIP = 4.0


def fit_normaliser(frame: pd.DataFrame, columns: list[str], prior: float = CHAMPION_PRIOR) -> dict:
    role = frame.groupby("position")[columns]
    champion = frame.groupby(["position", "champion_name"])[columns]
    return {
        "columns": columns,
        "prior": prior,
        "role_mean": role.mean().to_dict(orient="index"),
        "role_std": role.std().replace(0, np.nan).fillna(1.0).to_dict(orient="index"),
        "global_mean": frame[columns].mean().to_dict(),
        "global_std": frame[columns].std().replace(0, np.nan).fillna(1.0).to_dict(),
        "champion_mean": {
            f"{key[0]}|{key[1]}": value
            for key, value in champion.mean().to_dict(orient="index").items()
        },
        "champion_count": {
            f"{key[0]}|{key[1]}": int(value) for key, value in champion.size().items()
        },
    }


def player_residuals(
    frame: pd.DataFrame, columns: list[str], prior: float = CHAMPION_PRIOR
) -> pd.DataFrame:
    columns = [column for column in columns if column in frame.columns]
    data = frame[columns].astype(float)
    champion = frame["position"].astype(str) + "|" + frame["champion_name"].astype(str)
    mine = champion + "|" + frame["puuid"].astype(str)
    champion_n = champion.map(champion.value_counts()).astype(float)
    mine_n = mine.map(mine.value_counts()).astype(float)
    others_n = champion_n - mine_n
    others_sum = data.groupby(champion, observed=True).transform("sum") - data.groupby(
        mine, observed=True
    ).transform("sum")
    champion_mean = others_sum.div(others_n.where(others_n > 0), axis=0)
    role = data.groupby(frame["position"], observed=True)
    role_mean = role.transform("mean")
    spread = role.transform("std").replace(0, np.nan).fillna(1.0)
    weight = (others_n / (others_n + prior)).fillna(0.0)
    centre = champion_mean.fillna(role_mean).mul(weight, axis=0) + role_mean.mul(
        1.0 - weight, axis=0
    )
    return ((data - centre) / spread).clip(-CLIP, CLIP).fillna(0.0)


def apply_normaliser(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    columns = [column for column in stats["columns"] if column in frame.columns]
    prior = stats["prior"]
    roles = frame["position"].astype(str).to_numpy()
    champions = frame["champion_name"].astype(str).to_numpy()
    keys = [f"{role}|{champion}" for role, champion in zip(roles, champions)]
    role_mean = pd.DataFrame(
        [stats["role_mean"].get(role, stats["global_mean"]) for role in roles], index=frame.index
    )[columns]
    spread = pd.DataFrame(
        [stats["role_std"].get(role, stats["global_std"]) for role in roles], index=frame.index
    )[columns].replace(0, 1.0)
    champion_mean = pd.DataFrame(
        [stats["champion_mean"].get(key, stats["global_mean"]) for key in keys], index=frame.index
    )[columns]
    counts = np.array([float(stats["champion_count"].get(key, 0)) for key in keys])
    weight = (counts / (counts + prior))[:, None]
    centre = champion_mean.to_numpy() * weight + role_mean.to_numpy() * (1.0 - weight)
    scaled = (frame[columns].astype(float).to_numpy() - centre) / spread.to_numpy()
    return pd.DataFrame(scaled, index=frame.index, columns=columns).clip(-CLIP, CLIP).fillna(0.0)
