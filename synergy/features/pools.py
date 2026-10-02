import pandas as pd

from .positions import POSITIONS

PLAYER_CHAMPIONS = "player_champions.parquet"
TOP_CHAMPIONS = 3


def top_champions(seats: pd.DataFrame, keep: set[str] | None = None, top: int = TOP_CHAMPIONS) -> pd.DataFrame:
    seats = seats[seats["position"].isin(POSITIONS) & seats["champion_name"].notna()]
    if keep is not None:
        seats = seats[seats["puuid"].isin(keep)]
    last = seats["game_creation"] if "game_creation" in seats.columns else 0
    counted = (
        seats.assign(last=last)
        .groupby(["puuid", "position", "champion_name"])
        .agg(games=("puuid", "size"), last=("last", "max"))
        .reset_index()
        .sort_values(["puuid", "position", "games", "last", "champion_name"], ascending=[True, True, False, False, True])
    )
    kept = counted.groupby(["puuid", "position"]).head(top)
    return kept.groupby(["puuid", "position"], sort=False)["champion_name"].agg(",".join).rename("champions").reset_index()


def ranked_pool(champions: dict[str, int], top: int = TOP_CHAMPIONS) -> list[str]:
    return [name for name, _ in sorted(champions.items(), key=lambda item: -item[1])[:top]]
