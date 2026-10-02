import json

from synergy.api.artifacts import fetch_run
from synergy.config import Settings


class _S3:
    def __init__(self, objects):
        self.objects, self.downloads = objects, []

    def get_object(self, Bucket, Key):
        class _Body:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

        return {"Body": _Body(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        if ContinuationToken is None and len(keys) > 2:
            return {"Contents": [{"Key": key, "Size": len(self.objects[key])} for key in keys[:2]], "NextContinuationToken": "more"}
        rest = keys[2:] if ContinuationToken else keys
        return {"Contents": [{"Key": key, "Size": len(self.objects[key])} for key in rest]}

    def download_file(self, Bucket, Key, Filename):
        self.downloads.append(Key)
        with open(Filename, "wb") as handle:
            handle.write(self.objects[Key])


def _objects(run="20260927-032318"):
    return {
        "current.json": json.dumps({"run": run}).encode(),
        f"runs/{run}/manifest.json": b'{"id": "' + run.encode() + b'"}',
        f"runs/{run}/models/style_vectors.npz": b"vectors",
        f"runs/{run}/models/duo_scores.npz": b"scores",
        f"runs/{run}/models/seat_weights.npz": b"weights",
        f"runs/{run}/processed/player_profiles.parquet": b"profiles",
        f"runs/{run}/processed/player_names.parquet": b"names",
        f"runs/{run}/processed/propensity_report.json": b"{}",
        f"runs/{run}/processed/hinge.parquet": b"x" * 50,
        f"runs/{run}/processed/pair_history.parquet": b"x" * 60,
        f"runs/{run}/processed/player_styles.parquet": b"x" * 70,
        f"runs/{run}/evaluator/cells_rsp.npz": b"rsp",
        f"runs/{run}/evaluator/tendency_priors.json": b"{}",
        f"runs/{run}/evaluator/timeline_encoder.pt": b"x" * 80,
    }


def test_the_current_run_is_copied_from_the_bucket_without_the_big_tables(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path, model_store="s3://models")
    client = _S3(_objects())

    # when
    run = fetch_run(settings, client)

    # then
    assert run == "20260927-032318" and settings.served_run() == run
    assert (settings.served_model_dir / "style_vectors.npz").read_bytes() == b"vectors"
    assert (settings.served_processed_dir / "player_profiles.parquet").read_bytes() == b"profiles"
    assert (settings.runs_dir / run / "manifest.json").exists()
    assert not (settings.served_processed_dir / "hinge.parquet").exists() and not (settings.served_processed_dir / "player_styles.parquet").exists()
    evaluator = settings.served_model_dir.parent / "evaluator"
    assert (evaluator / "cells_rsp.npz").read_bytes() == b"rsp" and not (evaluator / "timeline_encoder.pt").exists()
    assert len(client.downloads) == 9


def test_a_second_fetch_copies_only_what_changed_and_follows_a_new_run(tmp_path):
    # given
    settings = Settings(data_dir=tmp_path, model_store="s3://models")
    objects = _objects()
    client = _S3(objects)
    fetch_run(settings, client)
    client.downloads.clear()

    # when
    objects["runs/20260927-032318/models/duo_scores.npz"] = b"scores v2"
    again = fetch_run(settings, client)
    objects.update(_objects("20261001-000000"))
    objects["current.json"] = json.dumps({"run": "20261001-000000"}).encode()
    moved = fetch_run(settings, client)

    # then
    assert again == "20260927-032318" and client.downloads[:1] == ["runs/20260927-032318/models/duo_scores.npz"]
    assert moved == "20261001-000000" and settings.served_run() == moved and (settings.served_model_dir / "style_vectors.npz").exists()
    assert len(client.downloads) == 1 + 9
