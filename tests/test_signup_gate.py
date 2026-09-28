import importlib.util
from pathlib import Path

import pytest

HANDLER = Path(__file__).resolve().parent.parent / "deploy" / "signup_gate" / "handler.py"
spec = importlib.util.spec_from_file_location("signup_gate", HANDLER)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class _Refused(Exception):
    pass


class _Counters:
    class exceptions:
        ConditionalCheckFailedException = _Refused

    def __init__(self):
        self.used = {}

    def update_item(self, TableName, Key, UpdateExpression, ConditionExpression, ExpressionAttributeValues):
        key = Key["sk"]["S"]
        limit = int(ExpressionAttributeValues[":limit"]["N"])
        if self.used.get(key, 0) >= limit:
            raise _Refused()
        self.used[key] = self.used.get(key, 0) + 1


def test_sign_ups_stop_at_the_daily_cap_and_reopen_the_next_day():
    # given
    counters = _Counters()

    # when
    today = [gate.admit(counters, "accounts", "2026-09-28", 3, 100, 0) for _ in range(4)]
    tomorrow = gate.admit(counters, "accounts", "2026-09-29", 3, 100, 0)

    # then
    assert today == [None, None, None, "Sign ups are full for today. Try again tomorrow."]
    assert tomorrow is None


def test_sign_ups_stop_for_good_at_the_total_cap():
    # given
    counters = _Counters()

    # when
    found = [gate.admit(counters, "accounts", f"2026-09-{day:02d}", 10, 2, 0) for day in (1, 2, 3)]

    # then
    assert found == [None, None, "Sign ups are closed for now."]


def test_only_self_sign_ups_are_counted_and_a_full_day_refuses_them(monkeypatch):
    # given
    counters = _Counters()
    monkeypatch.setattr(gate, "table", lambda: counters)
    monkeypatch.setenv("ACCOUNTS_TABLE", "accounts")
    monkeypatch.setenv("DAILY_SIGNUPS", "1")
    monkeypatch.setenv("TOTAL_SIGNUPS", "100")

    # when
    admin = gate.handler({"triggerSource": "PreSignUp_AdminCreateUser"}, None)
    first = gate.handler({"triggerSource": "PreSignUp_SignUp"}, None)
    with pytest.raises(Exception) as refused:
        gate.handler({"triggerSource": "PreSignUp_SignUp"}, None)

    # then
    assert admin == {"triggerSource": "PreSignUp_AdminCreateUser"} and first["triggerSource"] == "PreSignUp_SignUp"
    assert "full for today" in str(refused.value)
