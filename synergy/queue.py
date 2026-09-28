import importlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from celery import Celery, group
from celery.result import GroupResult
from kombu.exceptions import OperationalError

from .config import Settings, get_settings

JOBS = {
    "windows": "synergy.ingest.window:_window_chunk",
    "features": "synergy.features.build:_extract",
    "priority": "synergy.features.priority:_shard",
    "movement": "synergy.ml.movement:_shard",
    "reingest": "synergy.db.reingest:_chunk",
}
PRIVATE = ("riot_api_key",)
WAIT_SECONDS = 12 * 3600
PING_SECONDS = 3.0
POLL_SECONDS = 2.0
CONCURRENCY = 8
ERROR_TAIL = 4000


class WorkersMissing(RuntimeError):
    pass


def _app() -> Celery:
    settings = get_settings()
    found = Celery("synergy", broker=settings.broker_url, backend=settings.result_url)
    found.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        result_expires=WAIT_SECONDS,
        broker_transport_options={"visibility_timeout": WAIT_SECONDS},
        broker_connection_retry_on_startup=True,
        task_always_eager=settings.task_eager,
        task_eager_propagates=True,
    )
    return found


app = _app()


def settings_fields(settings: Settings) -> dict:
    return {
        name: str(value.resolve()) if isinstance(value, Path) else value
        for name, value in asdict(settings).items()
        if name not in PRIVATE
    }


def settings_from(values: dict) -> Settings:
    return Settings(**{**values, "data_dir": Path(values["data_dir"])})


def run_job(name: str, values: dict, args: list):
    module, attribute = JOBS[name].split(":")
    return getattr(importlib.import_module(module), attribute)(settings_from(values), *args)


def answer_request(request: Path, answer: Path) -> None:
    found = json.loads(request.read_text(encoding="utf-8"))
    answer.write_text(json.dumps(run_job(found["name"], found["values"], found["args"])), encoding="utf-8")


@app.task(name="synergy.job")
def job(name: str, values: dict, args: list):
    if app.conf.task_always_eager:
        return run_job(name, values, args)
    with tempfile.TemporaryDirectory() as folder:
        request, answer = Path(folder) / "request.json", Path(folder) / "answer.json"
        request.write_text(json.dumps({"name": name, "values": values, "args": args}), encoding="utf-8")
        done = subprocess.run([sys.executable, "-m", "synergy.queue", str(request), str(answer)], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"{name} job failed with exit {done.returncode}: {done.stderr[-ERROR_TAIL:]}")
        return json.loads(answer.read_text(encoding="utf-8"))


def require_workers() -> None:
    if app.conf.task_always_eager:
        return
    try:
        replies = app.control.ping(timeout=PING_SECONDS)
    except OperationalError as exc:
        raise WorkersMissing(f"The Celery broker at {app.conf.broker_url} is down. Start it with: docker compose up -d broker") from exc
    if not replies:
        raise WorkersMissing("No Celery workers are running. Start them with: python -m synergy worker")


def fan_out(name: str, settings: Settings, calls: list[tuple]) -> list:
    if name not in JOBS:
        raise KeyError(f"unknown job {name!r}, choose from {', '.join(JOBS)}")
    require_workers()
    values = settings_fields(settings)
    return collect(group([job.s(name, values, list(call)) for call in calls]).apply_async())


def collect(found: GroupResult) -> list:
    deadline = time.monotonic() + WAIT_SECONDS
    while not found.ready():
        if time.monotonic() > deadline:
            raise TimeoutError(f"jobs still running after {WAIT_SECONDS} seconds")
        time.sleep(POLL_SECONDS)
    failed = [result for result in found.results if result.failed()]
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(found.results)} jobs failed, the first with {failed[0].result!r}")
    values = [result.result for result in found.results]
    if not app.conf.task_always_eager:
        found.forget()
    return values


def start_worker(concurrency: int = CONCURRENCY) -> None:
    app.worker_main(["worker", "--loglevel=INFO", "--pool=threads", f"--concurrency={concurrency}"])


if __name__ == "__main__":
    answer_request(Path(sys.argv[1]), Path(sys.argv[2]))
