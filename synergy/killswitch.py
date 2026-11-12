import logging
import os
import re

import boto3

log = logging.getLogger("killswitch")
log.setLevel(logging.INFO)

ROLES = ("api", "history", "downloads", "sweep")
QUEUED = ("history", "downloads")
SERVICES = ("lambda", "pipes", "events", "ecs", "sns")
ACTIONS = ("trip", "restore", "status")
KEYS = (
    *(f"{role.upper()}_FUNCTION" for role in ROLES),
    *(f"{role.upper()}_MAPPING" for role in QUEUED),
    "EVALUATE_PIPE",
    "SWEEP_RULE",
    "JOBS_CLUSTER",
    "ALERTS_TOPIC",
    "API_CONCURRENCY",
)
COST = re.compile(r"ACTUAL (?:Amount: |Cost associated with this budget is )\$([0-9][0-9,]*(?:\.[0-9]+)?)")
PAUSED = "lolpredictor's backend is paused"
RUNNING = "lolpredictor's backend is running again"


def targets(environ) -> dict:
    return {
        "functions": {role: environ[f"{role.upper()}_FUNCTION"] for role in ROLES},
        "mappings": {role: environ[f"{role.upper()}_MAPPING"] for role in QUEUED},
        "pipe": environ["EVALUATE_PIPE"],
        "rule": environ["SWEEP_RULE"],
        "cluster": environ["JOBS_CLUSTER"],
        "topic": environ["ALERTS_TOPIC"],
        "api_concurrency": int(environ["API_CONCURRENCY"]),
    }


def local_targets(lam, name: str, region: str, account: str, environ) -> dict:
    found = {
        **{f"{role.upper()}_FUNCTION": f"{name}-{role}" for role in ROLES},
        "EVALUATE_PIPE": f"{name}-evaluate",
        "SWEEP_RULE": f"{name}-sweep",
        "JOBS_CLUSTER": name,
        "ALERTS_TOPIC": f"arn:aws:sns:{region}:{account}:{name}-alerts",
        "API_CONCURRENCY": "10",
        **{key: environ[key] for key in KEYS if environ.get(key)},
    }
    for role in QUEUED:
        key = f"{role.upper()}_MAPPING"
        if key not in found:
            mappings = lam.list_event_source_mappings(FunctionName=found[f"{role.upper()}_FUNCTION"]).get("EventSourceMappings", [])
            found[key] = mappings[0]["UUID"] if mappings else ""
    return targets(found)


def connect(region: str | None = None) -> dict:
    return {service: boto3.client(service, region_name=region) for service in SERVICES}


def _step(results: dict, step: str, action, *args) -> None:
    try:
        detail = action(*args)
    except Exception as problem:
        results[step] = {"ok": False, "detail": f"{type(problem).__name__}: {problem}"}
        log.error("%s failed: %s", step, results[step]["detail"])
    else:
        results[step] = {"ok": True, "detail": detail}
        log.info("%s %s", step, detail)


def _reserve(lam, function: str, count: int) -> str:
    lam.put_function_concurrency(FunctionName=function, ReservedConcurrentExecutions=count)
    return f"reserved {count}"


def _unreserve(lam, function: str) -> str:
    lam.delete_function_concurrency(FunctionName=function)
    return "unreserved"


def _concurrency(lam, function: str) -> str:
    count = lam.get_function_concurrency(FunctionName=function).get("ReservedConcurrentExecutions")
    return "unreserved" if count is None else f"reserved {count}"


def _mapping(lam, uuid: str) -> str:
    if not uuid:
        raise LookupError("no event source mapping")
    return lam.get_event_source_mapping(UUID=uuid)["State"]


def _switch_mapping(lam, uuid: str, enabled: bool) -> str:
    state = _mapping(lam, uuid)
    if state in (("Enabled", "Enabling") if enabled else ("Disabled", "Disabling")):
        return f"already {state.lower()}"
    lam.update_event_source_mapping(UUID=uuid, Enabled=enabled)
    return "enabling" if enabled else "disabling"


def _pipe(pipes, name: str) -> str:
    return pipes.describe_pipe(Name=name)["CurrentState"]


def _switch_pipe(pipes, name: str, running: bool) -> str:
    state = _pipe(pipes, name)
    if state in (("RUNNING", "STARTING") if running else ("STOPPED", "STOPPING")):
        return f"already {state.lower()}"
    (pipes.start_pipe if running else pipes.stop_pipe)(Name=name)
    return "starting" if running else "stopping"


def _rule(events, name: str) -> str:
    return events.describe_rule(Name=name)["State"]


def _switch_rule(events, name: str, enabled: bool) -> str:
    (events.enable_rule if enabled else events.disable_rule)(Name=name)
    return "enabled" if enabled else "disabled"


def _running(ecs, cluster: str) -> list[str]:
    tasks, token = [], None
    while True:
        page = ecs.list_tasks(cluster=cluster, desiredStatus="RUNNING", **({"nextToken": token} if token else {}))
        tasks += page.get("taskArns", [])
        token = page.get("nextToken")
        if not token:
            return tasks


def _stop_tasks(ecs, cluster: str) -> str:
    tasks = _running(ecs, cluster)
    for task in tasks:
        ecs.stop_task(cluster=cluster, task=task, reason=PAUSED)
    return f"stopped {len(tasks)}"


def _count_tasks(ecs, cluster: str) -> str:
    return f"{len(_running(ecs, cluster))} running"


def _publish(sns, topic: str, subject: str, opening: str, results: dict) -> str:
    lines = [opening, ""] + [f"{step}: {result['detail']}" for step, result in results.items()]
    sns.publish(TopicArn=topic, Subject=subject, Message="\n".join(lines))
    return "published"


def trip(clients: dict, names: dict, cost: str | None = None) -> dict:
    lam, results = clients["lambda"], {}
    for role, function in names["functions"].items():
        _step(results, f"{role}_concurrency", _reserve, lam, function, 0)
    for role, uuid in names["mappings"].items():
        _step(results, f"{role}_mapping", _switch_mapping, lam, uuid, False)
    _step(results, "evaluate_pipe", _switch_pipe, clients["pipes"], names["pipe"], False)
    _step(results, "sweep_rule", _switch_rule, clients["events"], names["rule"], False)
    _step(results, "jobs_tasks", _stop_tasks, clients["ecs"], names["cluster"])
    opening = f"{PAUSED} because this month's cost reached ${cost}." if cost else f"{PAUSED}."
    _step(results, "alerts", _publish, clients["sns"], names["topic"], PAUSED, opening, dict(results))
    return results


def restore(clients: dict, names: dict) -> dict:
    lam, results = clients["lambda"], {}
    for role, function in names["functions"].items():
        if role == "api":
            _step(results, "api_concurrency", _reserve, lam, function, names["api_concurrency"])
        else:
            _step(results, f"{role}_concurrency", _unreserve, lam, function)
    for role, uuid in names["mappings"].items():
        _step(results, f"{role}_mapping", _switch_mapping, lam, uuid, True)
    _step(results, "evaluate_pipe", _switch_pipe, clients["pipes"], names["pipe"], True)
    _step(results, "sweep_rule", _switch_rule, clients["events"], names["rule"], True)
    _step(results, "alerts", _publish, clients["sns"], names["topic"], RUNNING, f"{RUNNING}.", dict(results))
    return results


def status(clients: dict, names: dict) -> dict:
    lam, results = clients["lambda"], {}
    for role, function in names["functions"].items():
        _step(results, f"{role}_concurrency", _concurrency, lam, function)
    for role, uuid in names["mappings"].items():
        _step(results, f"{role}_mapping", _mapping, lam, uuid)
    _step(results, "evaluate_pipe", _pipe, clients["pipes"], names["pipe"])
    _step(results, "sweep_rule", _rule, clients["events"], names["rule"])
    _step(results, "jobs_tasks", _count_tasks, clients["ecs"], names["cluster"])
    return results


def parse(event: dict, budget: str | None = None) -> tuple[str, str | None]:
    records = event.get("Records") or []
    if records and "Sns" in records[0]:
        found = COST.search(records[0]["Sns"].get("Message") or "")
        return "trip", found.group(1) if found else budget
    action = event.get("action")
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    return action, event.get("cost")


def handler(event, context):
    action, cost = parse(event, os.environ.get("KILLSWITCH_BUDGET"))
    clients, names = connect(), targets(os.environ)
    if action == "trip":
        return trip(clients, names, cost)
    if action == "restore":
        return restore(clients, names)
    return status(clients, names)
