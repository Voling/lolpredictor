PLATFORM_TO_REGION = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
    "kr": "asia",
    "jp1": "asia",
    "eun1": "europe",
    "euw1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
}


class UnknownPlatform(ValueError):
    pass


def region_for(platform: str) -> str:
    key = platform.lower()
    if key not in PLATFORM_TO_REGION:
        raise UnknownPlatform(platform)
    return PLATFORM_TO_REGION[key]


def platform_host(platform: str) -> str:
    return f"https://{platform.lower()}.api.riotgames.com"


def region_host(region: str) -> str:
    return f"https://{region.lower()}.api.riotgames.com"


def split_riot_id(riot_id: str) -> tuple[str, str]:
    if "#" not in riot_id:
        raise ValueError(f"riot id must look like name#tag, got {riot_id!r}")
    name, _, tag = riot_id.partition("#")
    name = name.strip()
    tag = tag.strip()
    if not name or not tag:
        raise ValueError(f"riot id must look like name#tag, got {riot_id!r}")
    return name, tag


def join_riot_id(name: str, tag: str) -> str:
    return f"{name}#{tag}"
