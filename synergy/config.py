import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


TIER_ORDER = [
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]


@dataclass
class Settings:
    riot_api_key: str = field(default_factory=lambda: _str("RIOT_API_KEY", ""))
    platform: str = field(default_factory=lambda: _str("RIOT_PLATFORM", "na1"))
    region: str = field(default_factory=lambda: _str("RIOT_REGION", "americas"))
    seed_riot_id: str = field(default_factory=lambda: _str("SEED_RIOT_ID", "bblskibs#gotg"))

    queue_id: int = field(default_factory=lambda: _int("CRAWL_QUEUE_ID", 420))
    redis_url: str = field(default_factory=lambda: _str("REDIS_URL", "redis://localhost:6380/0"))
    cache_ttl: int = field(default_factory=lambda: _int("CACHE_TTL_SECONDS", 86400))
    min_tier: str = field(default_factory=lambda: _str("CRAWL_MIN_TIER", "DIAMOND"))
    max_tier: str = field(default_factory=lambda: _str("CRAWL_MAX_TIER", "MASTER"))
    matches_per_player: int = field(default_factory=lambda: _int("CRAWL_MATCHES_PER_PLAYER", 100))
    max_matches: int = field(default_factory=lambda: _int("CRAWL_MAX_MATCHES", 2000))
    max_players: int = field(default_factory=lambda: _int("CRAWL_MAX_PLAYERS", 400))
    max_depth: int = field(default_factory=lambda: _int("CRAWL_MAX_DEPTH", 3))
    fetch_timelines: bool = field(default_factory=lambda: _str("CRAWL_TIMELINES", "1") == "1")
    max_requests: int = field(default_factory=lambda: _int("CRAWL_MAX_REQUESTS", 2600))
    identify_reserve: int = field(default_factory=lambda: _int("CRAWL_IDENTIFY_RESERVE", 200))
    crawl_since_days: int = field(default_factory=lambda: _int("CRAWL_SINCE_DAYS", 0))
    crawl_discover: bool = field(default_factory=lambda: _bool("CRAWL_DISCOVER", True))
    apex_min_league_points: int = field(default_factory=lambda: _int("CRAWL_APEX_MIN_LP", 500))
    apex_tiers: str = field(default_factory=lambda: _str("CRAWL_APEX_TIERS", "challenger,grandmaster,master"))
    min_average_lp: int = field(default_factory=lambda: _int("CORPUS_MIN_AVERAGE_LP", 2800))
    min_ranked_participants: int = field(default_factory=lambda: _int("CORPUS_MIN_RANKED", 2))
    season_start: str = field(default_factory=lambda: _str("CORPUS_SEASON_START", "2026-01-01"))

    rate_short_requests: int = field(default_factory=lambda: _int("RIOT_RATE_SHORT_REQUESTS", 20))
    rate_short_seconds: float = field(default_factory=lambda: _float("RIOT_RATE_SHORT_SECONDS", 1.0))
    rate_long_requests: int = field(default_factory=lambda: _int("RIOT_RATE_LONG_REQUESTS", 100))
    rate_long_seconds: float = field(default_factory=lambda: _float("RIOT_RATE_LONG_SECONDS", 120.0))
    max_retries: int = field(default_factory=lambda: _int("RIOT_MAX_RETRIES", 4))
    concurrency: int = field(default_factory=lambda: _int("RIOT_CONCURRENCY", 8))

    data_dir: Path = field(default_factory=lambda: Path(_str("DATA_DIR", str(REPO_ROOT / "data"))))
    database_url: str = field(
        default_factory=lambda: _str(
            "DATABASE_URL", "postgresql://synergy:synergy@localhost:5432/synergy"
        )
    )

    shrinkage_k: float = field(default_factory=lambda: _float("SYNERGY_SHRINKAGE_K", 12.0))
    style_shrinkage_k: float = field(default_factory=lambda: _float("STYLE_SHRINKAGE_K", 5.0))
    min_profile_games: int = field(default_factory=lambda: _int("MIN_PROFILE_GAMES", 3))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def match_dir(self) -> Path:
        return self.raw_dir / "matches"

    @property
    def timeline_dir(self) -> Path:
        return self.raw_dir / "timelines"

    @property
    def window_dir(self) -> Path:
        return self.data_dir / "early"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def tier_floor(self) -> int:
        return TIER_ORDER.index(self.min_tier.upper())

    @property
    def tier_ceiling(self) -> int:
        return TIER_ORDER.index(self.max_tier.upper())

    def ensure_dirs(self) -> None:
        for path in (self.match_dir, self.timeline_dir, self.window_dir, self.processed_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
