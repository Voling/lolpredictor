import math
import random

from ..config import Settings, get_settings
from ..ranks import rank_to_lp
from .store import Store

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
CHAMPIONS = {
    "TOP": [(86, "Garen"), (122, "Darius"), (24, "Jax"), (58, "Renekton"), (54, "Malphite")],
    "JUNGLE": [(64, "LeeSin"), (60, "Elise"), (11, "MasterYi"), (79, "Gragas"), (76, "Nidalee")],
    "MIDDLE": [(103, "Ahri"), (238, "Zed"), (7, "Leblanc"), (134, "Syndra"), (61, "Orianna")],
    "BOTTOM": [(22, "Ashe"), (51, "Caitlyn"), (222, "Jinx"), (236, "Lucian"), (81, "Ezreal")],
    "UTILITY": [(412, "Thresh"), (117, "Lulu"), (43, "Karma"), (89, "Leona"), (497, "Rakan")],
}
ANCHORS = {
    "TOP": {100: (1900, 10500), 200: (3600, 13000)},
    "MIDDLE": {100: (6600, 6600), 200: (8600, 8600)},
    "BOTTOM": {100: (10500, 1900), 200: (13000, 3600)},
    "UTILITY": {100: (10500, 1900), 200: (13000, 3600)},
    "JUNGLE": {100: (5200, 8200), 200: (9800, 6800)},
}
OBJECTIVES = [(9800, 4400), (4400, 10200), (7500, 7500), (11000, 11000), (4000, 4000)]
STYLE_KEYS = ["aggro", "farm", "vision", "roam", "group", "objective"]
TIERS = [("DIAMOND", "II"), ("DIAMOND", "I"), ("MASTER", "I")]


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def true_synergy(a: dict, b: dict) -> float:
    return (
        0.5 * (a["roam"] * b["vision"] + b["roam"] * a["vision"])
        + 0.6 * a["group"] * b["group"]
        - 0.7 * abs(a["aggro"] - b["aggro"])
        - 0.4 * a["farm"] * b["farm"]
    )


def make_players(count: int, rng: random.Random, seed_riot_id: str) -> list[dict]:
    players = []
    for index in range(count):
        style = {key: rng.betavariate(2.2, 2.2) for key in STYLE_KEYS}
        tier, division = rng.choice(TIERS)
        name = seed_riot_id.split("#")[0] if index == 0 else f"crawler{index:03d}"
        tag = seed_riot_id.split("#")[1] if index == 0 else "NA1"
        players.append(
            {
                "puuid": f"synthetic-puuid-{index:04d}" + "0" * 60,
                "game_name": name,
                "tag_line": tag,
                "skill": rng.gauss(0, 1),
                "tier": tier,
                "division": division,
                "league_points": rng.randint(0, 99),
                "main": POSITIONS[index % 5],
                **style,
            }
        )
    return players


def _positions_for_minute(player: dict, role: str, team: int, minute: int, rng: random.Random,
                          objective: tuple[int, int]) -> tuple[float, float]:
    anchor = ANCHORS[role][team]
    roam_chance = 0.10 + 0.55 * player["roam"]
    group_chance = 0.25 + 0.6 * player["group"]
    if minute >= 14 and rng.random() < group_chance:
        base = objective
        spread = 900
    elif rng.random() < roam_chance:
        base = ANCHORS[rng.choice(POSITIONS)][team]
        spread = 1200
    else:
        base = anchor
        spread = 700
    x = min(max(rng.gauss(base[0], spread), 200.0), 14800.0)
    y = min(max(rng.gauss(base[1], spread), 200.0), 14800.0)
    return x, y


def _make_match(match_index: int, roster: list[dict], rng: random.Random, settings: Settings) -> tuple[dict, dict]:
    teams = {100: roster[:5], 200: roster[5:]}
    assignments: dict[int, list[dict]] = {}
    for team, members in teams.items():
        rng.shuffle(members)
        assignments[team] = [dict(player, role=POSITIONS[i]) for i, player in enumerate(members)]

    strength = {}
    synergy = {}
    for team, members in assignments.items():
        strength[team] = sum(player["skill"] for player in members) / math.sqrt(5)
        total = 0.0
        for i in range(5):
            for j in range(i + 1, 5):
                total += true_synergy(members[i], members[j])
        synergy[team] = total / 10.0
    edge = 0.9 * (strength[100] - strength[200]) + 6.0 * (synergy[100] - synergy[200])
    blue_wins = rng.random() < _logistic(edge)

    duration_min = int(min(max(rng.gauss(30, 6), 19), 46))
    match_id = f"NA1_SYN{match_index:06d}"
    game_creation = 1735689600000 + match_index * 2_700_000 + rng.randint(0, 900_000)

    participants = []
    pid = 0
    for team, members in assignments.items():
        for player in members:
            pid += 1
            participants.append((pid, team, player))

    frames = []
    positions: dict[int, list[tuple[float, float]]] = {pid: [] for pid, _, _ in participants}
    gold: dict[int, list[float]] = {pid: [] for pid, _, _ in participants}
    cs_track: dict[int, list[float]] = {pid: [] for pid, _, _ in participants}
    kills: dict[int, int] = {pid: 0 for pid, _, _ in participants}
    deaths: dict[int, int] = {pid: 0 for pid, _, _ in participants}
    assists: dict[int, int] = {pid: 0 for pid, _, _ in participants}
    wards: dict[int, int] = {pid: 0 for pid, _, _ in participants}

    objective_by_minute = [OBJECTIVES[rng.randrange(len(OBJECTIVES))] for _ in range(duration_min + 1)]
    for minute in range(duration_min + 1):
        participant_frames = {}
        for pid, team, player in participants:
            role = player["role"]
            point = _positions_for_minute(player, role, team, minute, rng, objective_by_minute[minute])
            positions[pid].append(point)
            winning = blue_wins == (team == 100)
            cs_rate = 3.4 + 3.6 * player["farm"] + (0.5 if role != "UTILITY" else -2.6)
            cs_value = max(0.0, cs_rate * minute + rng.gauss(0, 4))
            gold_value = (
                500
                + minute * (220 + 130 * player["farm"] + 60 * player["skill"])
                + kills[pid] * 300
                + assists[pid] * 130
                + (minute * 22 if winning else 0)
                + rng.gauss(0, 120)
            )
            cs_track[pid].append(cs_value)
            gold[pid].append(gold_value)
            damage_done = minute * (120 + 420 * player["aggro"]) + rng.gauss(0, 60)
            damage_taken = minute * (180 + 300 * player["aggro"]) + rng.gauss(0, 60)
            participant_frames[str(pid)] = {
                "participantId": pid,
                "damageStats": {
                    "totalDamageDoneToChampions": round(max(0.0, damage_done)),
                    "totalDamageTaken": round(max(0.0, damage_taken)),
                },
                "currentGold": max(0.0, rng.gauss(400, 200)),
                "totalGold": round(gold_value),
                "level": min(18, 1 + int(minute * 0.62)),
                "xp": round(minute * (330 + 90 * player["farm"])),
                "minionsKilled": round(cs_value * (1 - 0.35 * (role == "JUNGLE"))),
                "jungleMinionsKilled": round(cs_value * 0.35) if role == "JUNGLE" else 0,
                "position": {"x": round(point[0]), "y": round(point[1])},
            }
        events = []
        if minute > 2:
            expected = 0.5 + 0.05 * minute / 10
            for _ in range(int(rng.expovariate(1 / expected))):
                events.append(_kill_event(minute, participants, positions, rng, kills, deaths, assists))
        for pid, team, player in participants:
            if rng.random() < 0.15 + 0.6 * player["vision"]:
                wards[pid] += 1
                events.append(
                    {
                        "type": "WARD_PLACED",
                        "timestamp": minute * 60000 + rng.randint(0, 59000),
                        "creatorId": pid,
                        "wardType": "YELLOW_TRINKET",
                    }
                )
        if minute in (12, 20, 26, 33) and minute <= duration_min:
            team = 100 if rng.random() < 0.5 else 200
            members = [pid for pid, t, _ in participants if t == team]
            events.append(
                {
                    "type": "ELITE_MONSTER_KILL",
                    "timestamp": minute * 60000,
                    "killerId": rng.choice(members),
                    "killerTeamId": team,
                    "monsterType": "DRAGON",
                    "assistingParticipantIds": rng.sample(members, 2),
                }
            )
        frames.append(
            {
                "timestamp": minute * 60000,
                "participantFrames": participant_frames,
                "events": [event for event in events if event],
            }
        )

    info_participants = []
    for pid, team, player in participants:
        role = player["role"]
        champion_id, champion_name = CHAMPIONS[role][rng.randrange(5)]
        win = blue_wins == (team == 100)
        team_kills = sum(kills[other] for other, other_team, _ in participants if other_team == team) or 1
        final_gold = gold[pid][-1]
        final_cs = cs_track[pid][-1]
        info_participants.append(
            {
                "participantId": pid,
                "puuid": player["puuid"],
                "teamId": team,
                "teamPosition": role,
                "individualPosition": role,
                "championId": champion_id,
                "championName": champion_name,
                "win": win,
                "kills": kills[pid],
                "deaths": deaths[pid],
                "assists": assists[pid],
                "goldEarned": round(final_gold),
                "totalMinionsKilled": round(final_cs * (0.65 if role == "JUNGLE" else 1.0)),
                "neutralMinionsKilled": round(final_cs * 0.35) if role == "JUNGLE" else 0,
                "champExperience": round(duration_min * (330 + 90 * player["farm"])),
                "totalDamageDealtToChampions": round(
                    duration_min * (280 + 700 * player["aggro"] + 90 * player["skill"]) + rng.gauss(0, 900)
                ),
                "totalDamageTaken": round(duration_min * (450 + 500 * player["aggro"]) + rng.gauss(0, 900)),
                "damageDealtToObjectives": round(duration_min * (120 + 900 * player["objective"])),
                "damageDealtToTurrets": round(duration_min * (70 + 500 * player["objective"])),
                "visionScore": round(duration_min * (0.35 + 1.5 * player["vision"])),
                "wardsPlaced": wards[pid],
                "wardsKilled": round(wards[pid] * 0.35 * player["vision"] * 2),
                "detectorWardsPlaced": round(wards[pid] * 0.3 * player["vision"]),
                "timeCCingOthers": round(duration_min * (0.4 + 1.4 * player["group"])),
                "totalTimeSpentDead": round(deaths[pid] * (10 + duration_min * 0.6)),
                "turretTakedowns": round(rng.random() * 4 * player["objective"] + (2 if win else 0)),
                "dragonKills": round(rng.random() * 2 * player["objective"]),
                "baronKills": 1 if (win and rng.random() < 0.3) else 0,
                "firstBloodKill": False,
                "largestKillingSpree": max(0, round(kills[pid] * 0.4)),
                "gameEndedInSurrender": (not win) and rng.random() < 0.35,
                "gameEndedInEarlySurrender": False,
                "challenges": {
                    "killParticipation": min(1.0, (kills[pid] + assists[pid]) / team_kills),
                    "soloKills": round(kills[pid] * player["aggro"] * 0.4),
                    "laneMinionsFirst10Minutes": round(cs_track[pid][min(10, duration_min)] * 0.9),
                    "turretPlatesTaken": round(rng.random() * 3 * player["objective"]),
                    "saveAllyFromDeath": round(rng.random() * 3 * player["group"]),
                    "visionScoreAdvantageLaneOpponent": round(rng.gauss(0, 0.3), 3),
                },
            }
        )

    match = {
        "metadata": {
            "matchId": match_id,
            "participants": [player["puuid"] for _, _, player in participants],
        },
        "info": {
            "gameCreation": game_creation,
            "gameDuration": duration_min * 60,
            "gameVersion": "15.1.1.1",
            "queueId": settings.queue_id,
            "platformId": settings.platform.upper(),
            "participants": info_participants,
            "teams": [
                {"teamId": 100, "win": blue_wins},
                {"teamId": 200, "win": not blue_wins},
            ],
        },
    }
    timeline = {
        "metadata": {
            "matchId": match_id,
            "participants": [player["puuid"] for _, _, player in participants],
        },
        "info": {
            "frameInterval": 60000,
            "participants": [
                {"participantId": pid, "puuid": player["puuid"]} for pid, _, player in participants
            ],
            "frames": frames,
        },
    }
    return match, timeline


def _kill_event(minute, participants, positions, rng, kills, deaths, assists) -> dict | None:
    weights = [(pid, team, 0.2 + player["aggro"] + 0.4 * player["skill"]) for pid, team, player in participants]
    total = sum(max(w, 0.05) for _, _, w in weights)
    pick = rng.random() * total
    killer_id, killer_team = weights[0][0], weights[0][1]
    for pid, team, weight in weights:
        pick -= max(weight, 0.05)
        if pick <= 0:
            killer_id, killer_team = pid, team
            break
    victims = [pid for pid, team, _ in participants if team != killer_team]
    if not victims:
        return None
    victim_id = rng.choice(victims)
    killer_point = positions[killer_id][minute]
    allies = [
        (pid, math.hypot(positions[pid][minute][0] - killer_point[0], positions[pid][minute][1] - killer_point[1]))
        for pid, team, _ in participants
        if team == killer_team and pid != killer_id
    ]
    helpers = [pid for pid, distance in allies if distance < 2600 and rng.random() < 0.85]
    kills[killer_id] += 1
    deaths[victim_id] += 1
    for pid in helpers:
        assists[pid] += 1
    return {
        "type": "CHAMPION_KILL",
        "timestamp": minute * 60000 + rng.randint(0, 59000),
        "killerId": killer_id,
        "victimId": victim_id,
        "assistingParticipantIds": helpers,
        "position": {"x": round(killer_point[0]), "y": round(killer_point[1])},
        "victimDamageReceived": [
            {"participantId": pid, "physicalDamage": 100, "magicDamage": 0, "trueDamage": 0}
            for pid in [killer_id, *helpers]
        ],
        "victimDamageDealt": [
            {"participantId": victim_id, "physicalDamage": 60, "magicDamage": 0, "trueDamage": 0}
        ],
    }


def generate(
    matches: int = 400,
    players: int = 120,
    duos: int = 30,
    seed: int = 7,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    rng = random.Random(seed)
    roster = make_players(players, rng, settings.seed_riot_id)
    duo_pairs = [tuple(rng.sample(range(players), 2)) for _ in range(duos)]
    with Store(settings) as store:
        for player in roster:
            store.upsert_player(
                player["puuid"],
                game_name=player["game_name"],
                tag_line=player["tag_line"],
                tier=player["tier"],
                division=player["division"],
                league_points=player["league_points"],
                lp_value=rank_to_lp(player["tier"], player["division"], player["league_points"]),
                in_scope=True,
                depth=0,
            )
        for index in range(matches):
            chosen: list[int] = []
            if rng.random() < 0.55:
                left, right = duo_pairs[rng.randrange(len(duo_pairs))]
                chosen.extend([left, right])
            while len(chosen) < 10:
                candidate = rng.randrange(players)
                if candidate not in chosen:
                    chosen.append(candidate)
            if rng.random() < 0.4 and 0 not in chosen:
                chosen[rng.randrange(10)] = 0
            selection = [roster[i] for i in chosen]
            match, timeline = _make_match(index, selection, rng, settings)
            store.save_match(match)
            store.save_timeline(match["metadata"]["matchId"], timeline)
        return store.counts()
