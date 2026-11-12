import pytest

from synergy import killswitch

ENVIRON = {
    "API_FUNCTION": "lp-api",
    "HISTORY_FUNCTION": "lp-history",
    "DOWNLOADS_FUNCTION": "lp-downloads",
    "SWEEP_FUNCTION": "lp-sweep",
    "HISTORY_MAPPING": "uuid-history",
    "DOWNLOADS_MAPPING": "uuid-downloads",
    "EVALUATE_PIPE": "lp-evaluate",
    "SWEEP_RULE": "lp-sweep-rule",
    "JOBS_CLUSTER": "lp-cluster",
    "ALERTS_TOPIC": "arn:aws:sns:us-west-2:1:lp-alerts",
    "API_CONCURRENCY": "10",
}
BUDGET_MESSAGE = (
    "AWS Budget Notification November 12, 2026\nAWS Account 1\n\nDear AWS Customer,\n\n"
    "You requested that we alert you when the ACTUAL Cost associated with your lp-backend budget is greater than $30.00 for the current month. "
    "The ACTUAL Cost associated with this budget is $30.12.\n\n"
    "Budget Name: lp-backend\nBudget Type: Cost\nBudgeted Amount: $30.00\nAlert Type: ACTUAL\nAlert Threshold: > $30.00\nACTUAL Amount: $30.12\n"
)


class _Broken(Exception):
    pass


class _Lambda:
    def __init__(self, broken=()):
        self.broken = set(broken)
        self.reserved = {"lp-api": 10}
        self.states = {"uuid-history": "Enabled", "uuid-downloads": "Enabled"}
        self.calls = []

    def _check(self, name):
        if name in self.broken:
            raise _Broken(f"{name} is down")

    def put_function_concurrency(self, FunctionName, ReservedConcurrentExecutions):
        self._check(FunctionName)
        self.calls.append(("put", FunctionName, ReservedConcurrentExecutions))
        self.reserved[FunctionName] = ReservedConcurrentExecutions

    def delete_function_concurrency(self, FunctionName):
        self._check(FunctionName)
        self.calls.append(("delete", FunctionName))
        self.reserved.pop(FunctionName, None)

    def get_function_concurrency(self, FunctionName):
        return {"ReservedConcurrentExecutions": self.reserved[FunctionName]} if FunctionName in self.reserved else {}

    def get_event_source_mapping(self, UUID):
        self._check(UUID)
        return {"UUID": UUID, "State": self.states[UUID]}

    def update_event_source_mapping(self, UUID, Enabled):
        self.calls.append(("mapping", UUID, Enabled))
        self.states[UUID] = "Enabling" if Enabled else "Disabling"

    def list_event_source_mappings(self, FunctionName):
        return {"EventSourceMappings": [{"UUID": f"found-{FunctionName}"}]}


class _Pipes:
    def __init__(self, broken=False):
        self.state, self.broken, self.calls = "RUNNING", broken, []

    def describe_pipe(self, Name):
        if self.broken:
            raise _Broken("pipes is down")
        return {"Name": Name, "CurrentState": self.state}

    def stop_pipe(self, Name):
        self.calls.append(("stop", Name))
        self.state = "STOPPING"

    def start_pipe(self, Name):
        self.calls.append(("start", Name))
        self.state = "STARTING"


class _Events:
    def __init__(self):
        self.state, self.calls = "ENABLED", []

    def disable_rule(self, Name):
        self.calls.append(("disable", Name))
        self.state = "DISABLED"

    def enable_rule(self, Name):
        self.calls.append(("enable", Name))
        self.state = "ENABLED"

    def describe_rule(self, Name):
        return {"Name": Name, "State": self.state}


class _Ecs:
    def __init__(self, tasks):
        self.tasks, self.stopped = list(tasks), []

    def list_tasks(self, cluster, desiredStatus, nextToken=None):
        running, start = [task for task in self.tasks if (cluster, task) not in self.stopped], int(nextToken or 0)
        page = {"taskArns": running[start : start + 1]}
        if start + 1 < len(running):
            page["nextToken"] = str(start + 1)
        return page

    def stop_task(self, cluster, task, reason):
        self.stopped.append((cluster, task))


class _Sns:
    def __init__(self):
        self.published = []

    def publish(self, TopicArn, Subject, Message):
        self.published.append((TopicArn, Subject, Message))


def _clients(lam=None, pipes=None, tasks=("task-a", "task-b")):
    return {"lambda": lam or _Lambda(), "pipes": pipes or _Pipes(), "events": _Events(), "ecs": _Ecs(tasks), "sns": _Sns()}


def test_a_trip_pauses_each_named_part_and_says_what_the_month_cost():
    # given
    clients, names = _clients(), killswitch.targets(ENVIRON)

    # when
    results = killswitch.trip(clients, names, "30.12")

    # then
    assert all(result["ok"] for result in results.values())
    assert clients["lambda"].calls == [
        ("put", "lp-api", 0),
        ("put", "lp-history", 0),
        ("put", "lp-downloads", 0),
        ("put", "lp-sweep", 0),
        ("mapping", "uuid-history", False),
        ("mapping", "uuid-downloads", False),
    ]
    assert clients["pipes"].calls == [("stop", "lp-evaluate")] and clients["events"].calls == [("disable", "lp-sweep-rule")]
    assert clients["ecs"].stopped == [("lp-cluster", "task-a"), ("lp-cluster", "task-b")] and results["jobs_tasks"]["detail"] == "stopped 2"
    topic, subject, message = clients["sns"].published[0]
    assert topic == "arn:aws:sns:us-west-2:1:lp-alerts" and subject == "lolpredictor's backend is paused"
    assert message.startswith("lolpredictor's backend is paused because this month's cost reached $30.12.\n")


def test_a_failing_step_leaves_the_others_to_run_and_shows_in_the_summary():
    # given
    clients = _clients(lam=_Lambda(broken={"lp-history", "uuid-downloads"}), pipes=_Pipes(broken=True))

    # when
    results = killswitch.trip(clients, killswitch.targets(ENVIRON), "30")

    # then
    failed = {step for step, result in results.items() if not result["ok"]}
    assert failed == {"history_concurrency", "downloads_mapping", "evaluate_pipe"}
    assert clients["lambda"].reserved == {"lp-api": 0, "lp-downloads": 0, "lp-sweep": 0}
    assert clients["lambda"].states == {"uuid-history": "Disabling", "uuid-downloads": "Enabled"}
    assert clients["events"].state == "DISABLED" and len(clients["ecs"].stopped) == 2
    assert "history_concurrency: _Broken: lp-history is down" in clients["sns"].published[0][2]


def test_a_second_trip_changes_nothing_further():
    # given
    clients, names = _clients(), killswitch.targets(ENVIRON)
    killswitch.trip(clients, names, "30")
    before = list(clients["lambda"].calls)

    # when
    results = killswitch.trip(clients, names, "31")

    # then
    assert all(result["ok"] for result in results.values())
    assert clients["lambda"].calls[len(before) :] == [("put", name, 0) for name in ("lp-api", "lp-history", "lp-downloads", "lp-sweep")]
    assert results["history_mapping"]["detail"] == "already disabling" and results["evaluate_pipe"]["detail"] == "already stopping"
    assert clients["pipes"].calls == [("stop", "lp-evaluate")] and results["jobs_tasks"]["detail"] == "stopped 0"


def test_a_restore_puts_the_api_back_to_its_number_and_frees_the_rest():
    # given
    clients, names = _clients(), killswitch.targets(ENVIRON)
    killswitch.trip(clients, names, "30")

    # when
    results = killswitch.restore(clients, names)
    found = killswitch.status(clients, names)

    # then
    assert all(result["ok"] for result in results.values())
    assert clients["lambda"].reserved == {"lp-api": 10}
    assert clients["lambda"].states == {"uuid-history": "Enabling", "uuid-downloads": "Enabling"}
    assert clients["pipes"].calls[-1] == ("start", "lp-evaluate") and clients["events"].calls[-1] == ("enable", "lp-sweep-rule")
    assert clients["sns"].published[-1][2].startswith("lolpredictor's backend is running again.\n")
    assert {step: result["detail"] for step, result in found.items()} == {
        "api_concurrency": "reserved 10",
        "history_concurrency": "unreserved",
        "downloads_concurrency": "unreserved",
        "sweep_concurrency": "unreserved",
        "history_mapping": "Enabling",
        "downloads_mapping": "Enabling",
        "evaluate_pipe": "STARTING",
        "sweep_rule": "ENABLED",
        "jobs_tasks": "0 running",
    }


def test_a_budget_message_trips_with_its_cost_and_a_direct_call_names_its_action():
    # given
    event = {"Records": [{"EventSource": "aws:sns", "Sns": {"Message": BUDGET_MESSAGE}}]}
    bare = {"Records": [{"EventSource": "aws:sns", "Sns": {"Message": "budget reached"}}]}

    # when
    found = [killswitch.parse(event, "30"), killswitch.parse(bare, "30"), killswitch.parse({"action": "status"}), killswitch.parse({"action": "trip"})]

    # then
    assert found == [("trip", "30.12"), ("trip", "30"), ("status", None), ("trip", None)]
    with pytest.raises(ValueError):
        killswitch.parse({"action": "delete"})


def test_the_handler_trips_on_a_budget_message_with_names_from_its_environment(monkeypatch):
    # given
    clients = _clients()
    for key, value in {**ENVIRON, "KILLSWITCH_BUDGET": "30"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(killswitch, "connect", lambda region=None: clients)

    # when
    results = killswitch.handler({"Records": [{"EventSource": "aws:sns", "Sns": {"Message": BUDGET_MESSAGE}}]}, None)

    # then
    assert results["api_concurrency"] == {"ok": True, "detail": "reserved 0"}
    assert clients["sns"].published[0][2].startswith("lolpredictor's backend is paused because this month's cost reached $30.12.")


def test_local_names_come_from_the_environment_or_from_the_stack_name():
    # given
    environ = {"API_CONCURRENCY": "4", "EVALUATE_PIPE": "other-pipe", "HISTORY_QUEUE": "ignored"}

    # when
    names = killswitch.local_targets(_Lambda(), "lp", "us-west-2", "123", environ)

    # then
    assert names == {
        "functions": {"api": "lp-api", "history": "lp-history", "downloads": "lp-downloads", "sweep": "lp-sweep"},
        "mappings": {"history": "found-lp-history", "downloads": "found-lp-downloads"},
        "pipe": "other-pipe",
        "rule": "lp-sweep",
        "cluster": "lp",
        "topic": "arn:aws:sns:us-west-2:123:lp-alerts",
        "api_concurrency": 4,
    }
