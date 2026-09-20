import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings, get_settings

ROOT = Path(__file__).resolve().parent.parent
STEPS = ("window", "features", "stream", "profiles", "blocks", "mirrored", "scores")
COMMANDS = {
    "window": ("window", "--workers", "4"),
    "features": ("features", "--workers", "4"),
    "stream": ("stream",),
    "profiles": ("profiles",),
    "blocks": ("blocks",),
    "mirrored": ("mirrored",),
    "scores": ("interaction", "--skip-fit"),
    "network": ("pairnet",),
}
SERVED_MODELS = (
    "interaction_matrix.npz",
    "interaction_scores.npz",
    "seat_weights.npz",
    "seat_report.json",
    "interaction_report.json",
    "pairnet_report.json",
    "training_report.json",
    "synergy_model.pkl",
)
SERVED_PROCESSED = (
    "player_styles.parquet",
    "player_names.parquet",
    "hinge.parquet",
    "player_profiles.parquet",
    "pair_history.parquet",
    "propensity_report.json",
)
KEEP_RUNS = 5
MANIFEST = "manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plan(start: str | None = None, stop: str | None = None, network: bool = False) -> list[str]:
    steps = [*STEPS, *(["network"] if network else [])]
    for name in (start, stop):
        if name is not None and name not in steps:
            raise ValueError(f"unknown step {name!r}, choose from {', '.join(steps)}")
    first = steps.index(start) if start else 0
    last = steps.index(stop) + 1 if stop else len(steps)
    if last <= first:
        raise ValueError(f"{stop} comes before {start}")
    return steps[first:last]


def git_state(root: Path = ROOT) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"sha": sha, "dirty": bool(changed)}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def corpus_counts(settings: Settings) -> dict:
    from .ingest.store import Store

    try:
        store = Store(settings)
    except Exception as error:
        return {"error": str(error)[:200]}
    try:
        return {name: int(value) for name, value in store.counts().items()}
    finally:
        store.close()


def run_step(settings: Settings, step: str, log_path: Path, runner=None) -> int:
    command = [sys.executable, "-m", "synergy", *COMMANDS[step]]
    if runner is not None:
        return int(runner(step, command, log_path))
    env = {**os.environ, "DATA_DIR": str(settings.data_dir.resolve()), "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode


def write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _new_run(settings: Settings, prefix: str, counts) -> tuple[str, dict]:
    stamp = f"{prefix}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_id, tries = stamp, 1
    while (settings.runs_dir / run_id).exists():
        tries += 1
        run_id = f"{stamp}-{tries}"
    manifest = {"id": run_id, "started": _now(), "git": git_state(), "corpus": counts(settings), "steps": [], "status": "running"}
    write_manifest(settings.runs_dir / run_id, manifest)
    return run_id, manifest


def run_pipeline(
    settings: Settings | None = None,
    start: str | None = None,
    stop: str | None = None,
    network: bool = False,
    promote_after: bool = True,
    runner=None,
    counts=corpus_counts,
) -> dict:
    settings = settings or get_settings()
    steps = plan(start, stop, network)
    run_id, manifest = _new_run(settings, "", counts)
    run_dir = settings.runs_dir / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}: {' -> '.join(steps)}", flush=True)
    for step in steps:
        entry = {"name": step, "command": " ".join(COMMANDS[step]), "started": _now()}
        clock = time.time()
        code = run_step(settings, step, run_dir / "logs" / f"{step}.log", runner)
        entry.update({"seconds": round(time.time() - clock), "exit": code})
        manifest["steps"].append(entry)
        write_manifest(run_dir, manifest)
        print(f"  {step:9} exit {code} in {entry['seconds']:,}s", flush=True)
        if code != 0:
            manifest.update({"status": "failed", "finished": _now()})
            write_manifest(run_dir, manifest)
            return manifest
    manifest.update({"status": "built", "finished": _now()})
    write_manifest(run_dir, manifest)
    return promote(settings, run_id, counts=counts) if promote_after else manifest


def metrics(models: Path) -> dict:
    out = {}
    report = read_manifest_like(models / "interaction_report.json")
    if report:
        out["pair"] = {key: report.get(key) for key in ("matches", "cells_alone", "with_interaction", "gain", "interaction_spread", "informative")}
    seats = read_manifest_like(models / "seat_report.json")
    if seats:
        out["seats"] = seats
    network = read_manifest_like(models / "pairnet_report.json")
    if network:
        out["network"] = {"curved_gain": network.get("curved", {}).get("gain"), "interaction": network.get("interaction")}
    return out


def read_manifest_like(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def promote(settings: Settings | None = None, run_id: str | None = None, counts=corpus_counts) -> dict:
    settings = settings or get_settings()
    if run_id is None:
        run_id, manifest = _new_run(settings, "snapshot-", counts)
        manifest["status"] = "built"
    else:
        manifest = read_manifest(settings.runs_dir / run_id)
        if manifest is None:
            raise ValueError(f"no run named {run_id} under {settings.runs_dir}")
    run_dir = settings.runs_dir / run_id
    copied, missing = [], []
    for names, source, kind in ((SERVED_MODELS, settings.model_dir, "models"), (SERVED_PROCESSED, settings.processed_dir, "processed")):
        target = run_dir / "serve" / kind
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            if (source / name).exists():
                shutil.copy2(source / name, target / name)
                copied.append(f"{kind}/{name}")
            else:
                missing.append(f"{kind}/{name}")
    manifest["served"] = {"copied": copied, "missing": missing}
    manifest["metrics"] = metrics(run_dir / "serve" / "models")
    manifest.update({"status": "served", "promoted": _now()})
    write_manifest(run_dir, manifest)
    write_pointer(settings, run_id)
    prune(settings, current=run_id)
    return manifest


def write_pointer(settings: Settings, run_id: str) -> None:
    settings.pointer_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = settings.pointer_path.with_suffix(".tmp")
    scratch.write_text(json.dumps({"run": run_id, "promoted": _now()}), encoding="utf-8")
    os.replace(scratch, settings.pointer_path)


def list_runs(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    current = settings.served_run()
    found = []
    if settings.runs_dir.is_dir():
        for run_dir in sorted(settings.runs_dir.iterdir(), reverse=True):
            manifest = read_manifest(run_dir)
            if manifest:
                found.append({**manifest, "current": manifest["id"] == current})
    return found


def prune(settings: Settings, current: str | None, keep: int = KEEP_RUNS) -> list[str]:
    runs = [run["id"] for run in list_runs(settings)]
    removed = []
    for run_id in runs[keep:]:
        if run_id != current:
            shutil.rmtree(settings.runs_dir / run_id, ignore_errors=True)
            removed.append(run_id)
    return removed


def served_summary(settings: Settings | None = None) -> dict | None:
    settings = settings or get_settings()
    run_id = settings.served_run()
    manifest = read_manifest(settings.runs_dir / run_id) if run_id else None
    if manifest is None:
        return None
    return {
        "id": manifest["id"],
        "promoted": manifest.get("promoted"),
        "git": manifest.get("git"),
        "matches": (manifest.get("corpus") or {}).get("matches"),
        "metrics": manifest.get("metrics"),
    }
