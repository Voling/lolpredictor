import json
import logging
import threading

from fastapi import Body, FastAPI

from ..config import get_settings
from .accounts import DynamoTable
from .artifacts import bucket_of
from .downloads import GameStore, abandon_download, abandon_history, download, orchestrate, run_batch
from .evaluate import GIB, store_size
from .limiter import riot_client
from .sweep import sweep

_parts = None
_lock = threading.Lock()


class Parts:
    def __init__(self, settings):
        import boto3

        self.settings, self.sqs = settings, boto3.client("sqs")
        self.table = DynamoTable(settings.accounts_table)
        self.riot = riot_client(settings, self.table)
        self.store = GameStore(boto3.client("s3"), bucket_of(settings.evaluated_store), self.table, settings.evaluated_cap_gb * GIB)

    def send(self, bodies: list[dict]) -> None:
        entries = [{"Id": str(index), "MessageBody": json.dumps(body)} for index, body in enumerate(bodies)]
        found = self.sqs.send_message_batch(QueueUrl=self.settings.downloads_queue, Entries=entries)
        if found.get("Failed"):
            raise RuntimeError(f"{len(found['Failed'])} of {len(entries)} downloads were not queued")

    def launch(self, puuid: str) -> None:
        self.sqs.send_message(QueueUrl=self.settings.evaluate_queue, MessageBody=json.dumps({"puuid": puuid}))

    def request(self, puuid: str) -> None:
        self.sqs.send_message(QueueUrl=self.settings.history_queue, MessageBody=json.dumps({"puuid": puuid}))

    def defer(self, queue_url: str):
        def hold(record: dict, seconds: int) -> None:
            self.sqs.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=record["receiptHandle"], VisibilityTimeout=int(seconds))

        return hold

    def full(self) -> bool:
        return store_size(self.settings, self.table) >= self.store.cap_bytes


def get_parts() -> Parts:
    global _parts
    with _lock:
        if _parts is None:
            _parts = Parts(get_settings())
        return _parts


def history_batch(event: dict) -> dict:
    parts = get_parts()
    return run_batch(
        event.get("Records", []),
        lambda body, message_id: orchestrate(body["puuid"], parts.riot, parts.table, parts.store, parts.send, parts.launch, parts.full),
        lambda body, message_id, error: abandon_history(parts.table, body, error),
        parts.settings.history_retries,
        parts.table,
        parts.defer(parts.settings.history_queue),
    )


def downloads_batch(event: dict) -> dict:
    parts = get_parts()
    return run_batch(
        event.get("Records", []),
        lambda body, message_id: download(body, message_id, parts.riot, parts.table, parts.store, parts.launch),
        lambda body, message_id, error: abandon_download(parts.table, body, message_id, error, parts.launch),
        parts.settings.game_retries,
        parts.table,
        parts.defer(parts.settings.downloads_queue),
    )


def create_jobs_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="lolpredictor jobs", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/ready")
    def ready():
        return {"ok": True}

    @app.post("/history")
    def history(event: dict = Body(...)):
        return history_batch(event)

    @app.post("/downloads")
    def downloads(event: dict = Body(...)):
        return downloads_batch(event)

    @app.post("/sweep")
    def swept(event: dict = Body(...)):
        parts = get_parts()
        return sweep(parts.table, parts.request, parts.settings)

    return app


app = create_jobs_app()
