import pytest

from synergy.features.early import EARLY_FEATURE_COLUMNS, early_rows
from synergy.features.match import match_duration_minutes, participant_rows
from synergy.features.timeline import EARLY_MINUTES, PAIR_TIMELINE_COLUMNS, VISION_WARDS, pair_rows
from synergy.ingest.window import truncate_timeline

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
BLUE_JUNGLE = (5200, 8200)
RED_JUNGLE = (10500, 7000)
MID = (7500, 7500)


def build_match(duration: int = 1800) -> dict:
    participants = [
        {
            "participantId": index + 1,
            "puuid": f"p{index}",
            "teamId": 100 if index < 5 else 200,
            "teamPosition": ROLES[index % 5],
            "championId": 1,
            "championName": "Annie" if index < 5 else "Ashe",
            "win": index < 5,
            "gameEndedInSurrender": False,
        }
        for index in range(10)
    ]
    return {
        "metadata": {"matchId": "NA1_1", "participants": [f"p{i}" for i in range(10)]},
        "info": {
            "gameCreation": 1735689600000,
            "gameDuration": duration,
            "gameVersion": "16.16.1.1",
            "queueId": 420,
            "participants": participants,
        },
    }


def build_timeline(minutes: int = 25, blue_top_position=MID, events=None) -> dict:
    frames = []
    for minute in range(minutes):
        participant_frames = {}
        for index in range(10):
            position = blue_top_position if index == 0 else (MID if index < 5 else RED_JUNGLE)
            participant_frames[str(index + 1)] = {
                "participantId": index + 1,
                "totalGold": 500 * minute,
                "xp": 300 * minute,
                "minionsKilled": 6 * minute,
                "jungleMinionsKilled": 4 * minute if index % 5 == 1 else 0,
                "level": 1 + minute // 2,
                "position": {"x": position[0], "y": position[1]},
                "damageStats": {
                    "totalDamageDoneToChampions": 400 * minute,
                    "totalDamageTaken": 300 * minute,
                },
            }
        frames.append(
            {
                "timestamp": minute * 60000,
                "participantFrames": participant_frames,
                "events": [event for event in (events or []) if event["timestamp"] // 60000 == minute],
            }
        )
    return {
        "metadata": {"matchId": "NA1_1", "participants": [f"p{i}" for i in range(10)]},
        "info": {
            "frameInterval": 60000,
            "participants": [{"participantId": i + 1, "puuid": f"p{i}"} for i in range(10)],
            "frames": frames,
        },
    }


def test_duration_accepts_seconds_and_milliseconds():
    assert match_duration_minutes({"gameDuration": 1800}) == pytest.approx(30.0)
    assert match_duration_minutes({"gameDuration": 1800000}) == pytest.approx(30.0)
    assert match_duration_minutes({"gameDuration": 0}) == 1.0


def test_participant_rows_carry_match_metadata_only():
    rows = participant_rows(build_match())
    assert len(rows) == 10
    assert rows[0]["position"] == "TOP"
    assert rows[0]["patch"] == "16.16"
    assert rows[0]["win"] == 1
    assert rows[5]["win"] == 0
    assert not any(key.startswith("e_") for key in rows[0])


def test_truncate_keeps_only_the_first_window():
    events = [
        {"type": "WARD_PLACED", "timestamp": 60000, "creatorId": 1, "wardType": "SIGHT_WARD"},
        {"type": "WARD_PLACED", "timestamp": 1200000, "creatorId": 1, "wardType": "SIGHT_WARD"},
    ]
    truncated = truncate_timeline(build_timeline(25, events=events), minutes=15)
    frames = truncated["info"]["frames"]
    assert len(frames) == 16
    assert max(frame["timestamp"] for frame in frames) == 15 * 60000
    kept = [event for frame in frames for event in frame["events"]]
    assert len(kept) == 1
    assert truncated["info"]["windowMinutes"] == 15


def test_truncate_drops_late_events_inside_a_kept_frame():
    late = {"type": "WARD_PLACED", "timestamp": 15 * 60000 + 5000, "creatorId": 1, "wardType": "SIGHT_WARD"}
    timeline = build_timeline(25)
    timeline["info"]["frames"][15]["events"].append(late)
    truncated = truncate_timeline(timeline, minutes=15)
    assert all(not frame["events"] for frame in truncated["info"]["frames"])


def test_early_rows_cover_every_participant_and_are_finite():
    rows = early_rows(build_match(), build_timeline())
    assert len(rows) == 10
    for row in rows:
        for column in EARLY_FEATURE_COLUMNS:
            assert column in row
            assert row[column] == row[column]


def test_enemy_jungle_presence_is_measured_from_the_players_own_side():
    rows = {row["puuid"]: row for row in early_rows(build_match(), build_timeline(blue_top_position=RED_JUNGLE))}
    assert rows["p0"]["e_enemy_jungle_share"] > 0.9
    assert rows["p5"]["e_enemy_jungle_share"] == 0.0
    assert rows["p5"]["e_own_jungle_share"] > 0.9


def test_invade_camp_credit_needs_both_position_and_jungle_farm():
    laner = {row["puuid"]: row for row in early_rows(build_match(), build_timeline(blue_top_position=RED_JUNGLE))}
    assert laner["p0"]["e_invade_cs_share"] == 0.0


def test_trade_ratio_uses_damage_given_over_the_exchange():
    rows = {row["puuid"]: row for row in early_rows(build_match(), build_timeline())}
    assert rows["p0"]["e_trade_ratio"] == pytest.approx(400 / 700, abs=1e-6)


def test_ward_features_count_vision_wards_only():
    events = [
        {"type": "WARD_PLACED", "timestamp": 60000, "creatorId": 1, "wardType": "YELLOW_TRINKET"},
        {"type": "WARD_PLACED", "timestamp": 61000, "creatorId": 1, "wardType": "CONTROL_WARD"},
        {"type": "WARD_PLACED", "timestamp": 62000, "creatorId": 1, "wardType": "TEEMO_MUSHROOM"},
        {"type": "WARD_PLACED", "timestamp": 63000, "creatorId": 1, "wardType": "UNDEFINED"},
    ]
    rows = {row["puuid"]: row for row in early_rows(build_match(), build_timeline(events=events))}
    assert rows["p0"]["e_wards_pm"] == pytest.approx(2 / 15)
    assert rows["p0"]["e_control_wards_pm"] == pytest.approx(1 / 15)
    assert "TEEMO_MUSHROOM" not in VISION_WARDS
    assert "UNDEFINED" not in VISION_WARDS


def test_first_back_is_the_first_purchase_after_the_start():
    events = [
        {"type": "ITEM_PURCHASED", "timestamp": 1000, "participantId": 1, "itemId": 1055},
        {"type": "ITEM_PURCHASED", "timestamp": 240000, "participantId": 1, "itemId": 3006},
        {"type": "ITEM_PURCHASED", "timestamp": 240500, "participantId": 1, "itemId": 3006},
    ]
    rows = {row["puuid"]: row for row in early_rows(build_match(), build_timeline(events=events))}
    assert rows["p0"]["e_first_back_minute"] == pytest.approx(4.0)
    assert "e_purchases" not in rows["p0"]


def test_pair_rows_are_capped_at_the_early_window():
    rows = pair_rows(build_match(), build_timeline(40))
    assert len(rows) == 20
    for row in rows:
        for column in PAIR_TIMELINE_COLUMNS:
            assert column in row
        assert 0.0 <= row["pair_close_share"] <= 1.0
    assert EARLY_MINUTES == 15


def test_bot_lane_pairs_stand_closer_than_cross_map_pairs(corpus_settings):
    from synergy.ingest.store import Store

    lane, split = [], []
    with Store(corpus_settings) as store:
        for match_id in store.match_ids()[:10]:
            match = store.load_match(match_id)
            window = store.load_window(match_id)
            roles = {p["puuid"]: p["teamPosition"] for p in match["info"]["participants"]}
            for row in pair_rows(match, window):
                pairing = {roles[row["puuid_a"]], roles[row["puuid_b"]]}
                if pairing == {"BOTTOM", "UTILITY"}:
                    lane.append(row["pair_lane_distance"])
                elif pairing == {"TOP", "BOTTOM"}:
                    split.append(row["pair_lane_distance"])
    assert sum(lane) / len(lane) < sum(split) / len(split)


def test_position_at_snaps_to_the_nearest_frame_not_a_midpoint():
    from synergy.features.timeline import ParsedTimeline

    parsed = ParsedTimeline(build_match(), build_timeline())
    pid = next(iter(parsed.pid_to_puuid))
    track = parsed.positions[pid]
    assert parsed.position_at(pid, 4.0) == track[4]
    assert parsed.position_at(pid, 4.4) == track[4]
    assert parsed.position_at(pid, 4.6) == track[5]
    assert parsed.position_at(pid, 900.0) == track[-1]


def test_wave_state_ignores_jungle_camps_and_dead_minutes():
    from synergy.features.timeline import ParsedTimeline
    from synergy.features.wave import OFF_LANE, wave_states

    parsed = ParsedTimeline(build_match(), build_timeline())
    for pid in parsed.pid_to_puuid:
        states = wave_states(parsed, pid, 15)
        for minute, state in enumerate(states):
            if not parsed.alive_at(pid, float(minute)):
                assert state == OFF_LANE
