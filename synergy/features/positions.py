POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
UNKNOWN = "UNKNOWN"
KEY = ["puuid", "position"]
ALIASES = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "jg": "JUNGLE",
    "jng": "JUNGLE",
    "jgl": "JUNGLE",
    "mid": "MIDDLE",
    "middle": "MIDDLE",
    "bot": "BOTTOM",
    "bottom": "BOTTOM",
    "adc": "BOTTOM",
    "marksman": "BOTTOM",
    "support": "UTILITY",
    "supp": "UTILITY",
    "sup": "UTILITY",
    "utility": "UTILITY",
}


def parse_position(text: str) -> str:
    key = str(text).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"unknown position {text!r}, use one of top, jungle, mid, bot, support")
    return ALIASES[key]


def combination(left: str, right: str) -> str:
    return "+".join(sorted([left, right]))
