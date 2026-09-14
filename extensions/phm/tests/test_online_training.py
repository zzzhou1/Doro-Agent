import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from phm_mcp.admin import AdminService
from phm_mcp.inference import InferenceService
from phm_mcp.registry import REQUIRED_ARTIFACTS
from phm_mcp.training import resolve_device
from phm_mcp.worker import TrainingWorker


def _write_processed(path: Path) -> None:
    rng = np.random.default_rng(7)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_x=rng.normal(size=(8, 3, 2)).astype(np.float32),
        train_y=np.linspace(8, 1, 8, dtype=np.float32),
        validation_x=rng.normal(size=(4, 3, 2)).astype(np.float32),
        validation_y=np.linspace(4, 1, 4, dtype=np.float32),
        test_x=rng.normal(size=(2, 3, 2)).astype(np.float32),
        test_y=np.asarray([3, 2], dtype=np.float32),
        test_unit_ids=np.asarray([1, 2], dtype=np.int64),
        test_last_cycles=np.asarray([10, 12], dtype=np.int64),
        selected_feature_indices=np.asarray([0, 1], dtype=np.int64),
        feature_mean=np.zeros(2, dtype=np.float32),
        feature_scale=np.ones(2, dtype=np.float32),
        window_size=np.asarray(3),
    )


def _seed_v1(candidate: Path, target: Path) -> None:
    shutil.copytree(candidate, target)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["version"] = "v1"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert all((target / name).is_file() for name in REQUIRED_ARTIFACTS)


def test_online_job_train_promote_predict_and_rollback(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    runtime_dir = tmp_path / "runtime"
    _write_processed(data_dir / "processed" / "fd001.npz")
    admin = AdminService(data_dir, artifact_dir, runtime_dir)
    job = admin.submit_training_job(
        model="lstm",
        version="v2",
        epochs=1,
        batch_size=4,
        patience=1,
        device="cpu",
    )

    completed = TrainingWorker(data_dir, artifact_dir, runtime_dir).run_once()
    assert completed is not None
    assert completed["id"] == job["id"]
    assert completed["status"] == "succeeded"
    assert completed["progress"]["epoch"] == 1
    candidate = Path(completed["candidate_path"])
    assert all((candidate / name).is_file() for name in REQUIRED_ARTIFACTS)

    _seed_v1(candidate, artifact_dir / "lstm" / "v1")
    promoted = admin.promote_candidate_model(job["id"])
    assert promoted["registry"]["active"]["lstm"] == "v2"

    prediction = InferenceService(data_dir, artifact_dir).predict_rul(1, "lstm")
    assert prediction["version"] == "v2"
    assert prediction["predicted_rul"] >= 0

    rolled_back = admin.rollback_model("lstm")
    assert rolled_back["active"]["lstm"] == "v1"


def test_device_validation_on_cpu_build() -> None:
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises((RuntimeError, ValueError)):
        resolve_device("cuda:9999")
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("gpu")
