import json
import logging
import time
from pathlib import Path

from ..config import Settings

logger = logging.getLogger(__name__)
PROCESSED = ("player_names.parquet", "player_profiles.parquet", "propensity_report.json", "player_champions.parquet")
EVALUATOR = ("cells_prio.npz", "cells_rsp.npz", "cells_obj.npz", "cells_ward.npz", "cells_jgl.npz", "cells_jgl_sides.npz", "standardise.npz", "tendency_priors.json")
MANIFEST = "manifest.json"


def bucket_of(store: str) -> str:
    return store.removeprefix("s3://").split("/", 1)[0]


def current_run(client, bucket: str) -> str:
    body = client.get_object(Bucket=bucket, Key="current.json")["Body"].read()
    return str(json.loads(body)["run"])


def _listed(client, bucket: str, prefix: str) -> dict[str, int]:
    found = {}
    token = None
    while True:
        request = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        page = client.list_objects_v2(**request)
        for item in page.get("Contents", []):
            found[item["Key"]] = int(item["Size"])
        token = page.get("NextContinuationToken")
        if not token:
            return found


def _wanted(client, bucket: str, run: str) -> dict[str, int]:
    models = _listed(client, bucket, f"runs/{run}/models/")
    processed = {key: size for key, size in _listed(client, bucket, f"runs/{run}/processed/").items() if key.rsplit("/", 1)[-1] in PROCESSED}
    evaluator = {key: size for key, size in _listed(client, bucket, f"runs/{run}/evaluator/").items() if key.rsplit("/", 1)[-1] in EVALUATOR}
    manifest = _listed(client, bucket, f"runs/{run}/{MANIFEST}")
    return {**models, **processed, **evaluator, **manifest}


def _target(settings: Settings, run: str, key: str) -> Path:
    rest = key.split(f"runs/{run}/", 1)[1]
    if rest == MANIFEST:
        return settings.runs_dir / run / MANIFEST
    return settings.runs_dir / run / "serve" / rest


def fetch_run(settings: Settings, client=None) -> str:
    if client is None:
        import boto3

        client = boto3.client("s3")
    bucket = bucket_of(settings.model_store)
    clock = time.time()
    run = current_run(client, bucket)
    wanted = _wanted(client, bucket, run)
    copied = 0
    for key, size in wanted.items():
        target = _target(settings, run, key)
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.with_name(target.name + ".part")
        client.download_file(bucket, key, str(scratch))
        scratch.replace(target)
        copied += 1
    settings.pointer_path.parent.mkdir(parents=True, exist_ok=True)
    settings.pointer_path.write_text(json.dumps({"run": run}), encoding="utf-8")
    logger.info("run %s ready from %s, %d of %d files copied in %.1fs", run, bucket, copied, len(wanted), time.time() - clock)
    return run
