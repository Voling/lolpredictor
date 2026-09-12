from .timeline import ParsedTimeline

SPAN = 15


def _damage(entries: list[dict] | None) -> float:
    return sum(
        float(entry.get("magicDamage", 0.0))
        + float(entry.get("physicalDamage", 0.0))
        + float(entry.get("trueDamage", 0.0))
        for entry in entries or []
    )


def extra_rows(
    match: dict, timeline: dict, span: int = SPAN, parsed: ParsedTimeline | None = None
) -> list[dict]:
    parsed = parsed or ParsedTimeline(match, timeline)
    if len(parsed.minutes) < 8:
        return []
    limit = min(len(parsed.minutes) - 1, span)
    frames = (timeline.get("info", {}).get("frames") or [])[: limit + 1]
    if not frames:
        return []
    minutes = float(max(limit, 1))
    events = [
        event for event in parsed.events if float(event.get("timestamp", 0)) / 60000.0 <= limit
    ]
    plates: dict[int, int] = {pid: 0 for pid in parsed.pid_to_puuid}
    first_blood: dict[int, float] = {}
    bounty: dict[int, float] = {pid: 0.0 for pid in parsed.pid_to_puuid}
    given_up: dict[int, float] = {pid: 0.0 for pid in parsed.pid_to_puuid}
    streak: dict[int, float] = {pid: 0.0 for pid in parsed.pid_to_puuid}
    fought_back: dict[int, float] = {pid: 0.0 for pid in parsed.pid_to_puuid}
    deaths: dict[int, int] = {pid: 0 for pid in parsed.pid_to_puuid}
    for event in events:
        kind = event.get("type")
        if kind == "TURRET_PLATE_DESTROYED":
            pid = int(event.get("killerId", 0))
            if pid in plates:
                plates[pid] += 1
        elif kind == "CHAMPION_SPECIAL_KILL" and event.get("killType") == "KILL_FIRST_BLOOD":
            pid = int(event.get("killerId", 0))
            if pid in parsed.pid_to_puuid:
                first_blood.setdefault(pid, float(event.get("timestamp", 0)) / 60000.0)
        elif kind == "CHAMPION_KILL":
            killer = int(event.get("killerId", 0))
            victim = int(event.get("victimId", 0))
            if killer in bounty:
                bounty[killer] += float(event.get("bounty", 0.0))
                streak[killer] = max(streak[killer], float(event.get("killStreakLength", 0.0)))
            if victim in given_up:
                deaths[victim] += 1
                given_up[victim] += float(event.get("shutdownBounty", 0.0))
                fought_back[victim] += _damage(event.get("victimDamageDealt"))
    rows = []
    for pid, puuid in parsed.pid_to_puuid.items():
        key = str(pid)
        payloads = [
            frame["participantFrames"][key]
            for frame in frames
            if key in (frame.get("participantFrames") or {})
        ]
        if not payloads:
            continue
        last = payloads[-1]
        damage = last.get("damageStats") or {}
        done = float(damage.get("totalDamageDoneToChampions", 0.0))
        taken = float(damage.get("totalDamageTaken", 0.0))
        speeds = [float((p.get("championStats") or {}).get("movementSpeed", 0.0)) for p in payloads]
        rows.append(
            {
                "match_id": parsed.match_id,
                "puuid": puuid,
                "x_time_controlling_pm": float(last.get("timeEnemySpentControlled", 0.0)) / 1000.0 / minutes,
                "x_gold_per_second": float(last.get("goldPerSecond", 0.0)),
                "x_movement_speed": sum(speeds) / len(speeds) if speeds else 0.0,
                "x_plates": float(plates.get(pid, 0)),
                "x_took_first_blood": float(pid in first_blood),
                "x_first_blood_minute": first_blood.get(pid, 0.0),
                "x_bounty_earned": bounty.get(pid, 0.0),
                "x_bounty_given_up": given_up.get(pid, 0.0),
                "x_best_kill_streak": streak.get(pid, 0.0),
                "x_fought_back_per_death": fought_back.get(pid, 0.0) / max(deaths.get(pid, 0), 1),
                "x_physical_share": float(damage.get("physicalDamageDoneToChampions", 0.0)) / done if done else 0.0,
                "x_true_share": float(damage.get("trueDamageDoneToChampions", 0.0)) / done if done else 0.0,
                "x_taken_physical_share": float(damage.get("physicalDamageTaken", 0.0)) / taken if taken else 0.0,
                "x_wasted_damage": 1.0 - (done / float(damage.get("totalDamageDone", 0.0)))
                if float(damage.get("totalDamageDone", 0.0))
                else 0.0,
            }
        )
    return rows


EXTRA_FEATURE_COLUMNS = [
    "x_time_controlling_pm",
    "x_gold_per_second",
    "x_movement_speed",
    "x_plates",
    "x_took_first_blood",
    "x_first_blood_minute",
    "x_bounty_earned",
    "x_bounty_given_up",
    "x_best_kill_streak",
    "x_fought_back_per_death",
    "x_physical_share",
    "x_true_share",
    "x_taken_physical_share",
    "x_wasted_damage",
]
