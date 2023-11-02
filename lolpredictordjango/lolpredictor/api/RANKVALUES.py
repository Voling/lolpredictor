rank_values = {
        "IRON": 0,
        "BRONZE": 400,
        "SILVER": 800,
        "GOLD": 1200,
        "PLATINUM": 1600,
        "EMERALD": 2000,
        "DIAMOND": 2400,
        "MASTER": 2800,
        "GRANDMASTER": 2800,
        "CHALLENGER": 2800
    }
def rank_to_lp(tier: str, division: str, lp: int) -> int:
    base_lp = rank_values.get(tier.upper(), 0)
    division_lp = (4 - romanToInt(division)) * 100
    return base_lp + division_lp + lp

def romanToInt(s: str) -> int:
    romanToIntMap = {
        'I': 1,
        'V': 5,
    }
    total = 0
    prev_value = 0
    for char in s[::-1]:
        value = romanToIntMap[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total
