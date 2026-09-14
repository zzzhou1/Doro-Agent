from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .jobs import VERSION_PATTERN, JobStore
from .registry import MODEL_NAMES, ModelRegistry
from .supervisor import WorkerSupervisor

DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(?::[0-9]+)?|mps)$")
TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "epochs": 1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "patience": 1,
        "seed": 42,
        "device": "auto",
        "amp": False,
    },
    "standard": {
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "patience": 5,
        "seed": 42,
        "device": "auto",
        "amp": False,
    },
    "thorough": {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "patience": 8,
        "seed": 42,
        "device": "auto",
        "amp": False,
    },
}


class AdminService:
    """Validated control plane for queued training and model promotion."""

    def __init__(self, data_dir: Path, artifact_dir: Path, runtime_dir: Path) -> None:
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.runtime_dir = runtime_dir
        self.jobs = JobStore(runtime_dir / "jobs.sqlite")
        self.registry = ModelRegistry(artifact_dir)
        self.supervisor = WorkerSupervisor(data_dir, artifact_dir, runtime_dir)

    def submit_training_job(
        self,
        *,
        model: str,
        version: str | None = None,
        preset: str = "standard",
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        patience: int | None = None,
        seed: int | None = None,
        device: str | None = None,
        amp: bool | None = None,
        auto_start_worker: bool = True,
        preset_confirmed: bool = False,
    ) -> dict[str, Any]:
        self._validate_model(model)
        if preset not in TRAINING_PRESETS:
            raise ValueError("preset must be 'smoke', 'standard', or 'thorough'")
        overrides = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
            "device": device,
            "amp": amp,
        }
        has_overrides = any(value is not None for value in overrides.values())
        if not has_overrides and not preset_confirmed:
            resolved = TRAINING_PRESETS[preset]
            raise ValueError(
                "No hyperparameters were provided. Ask the user to confirm the "
                f"'{preset}' preset before submitting: {resolved}"
            )
        params = dict(TRAINING_PRESETS[preset])
        params.update({name: value for name, value in overrides.items() if value is not None})
        if epochs is not None and patience is None:
            params["patience"] = min(int(params["patience"]), epochs)
        parameter_source = "confirmed_preset" if not has_overrides else "preset_with_overrides"
        version = version or f"candidate-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        self._validate_version(version)
        self._validate_training_params(
            int(params["epochs"]),
            int(params["batch_size"]),
            float(params["learning_rate"]),
            int(params["patience"]),
            int(params["seed"]),
            str(params["device"]),
            bool(params["amp"]),
        )
        processed = self.data_dir / "processed" / "fd001.npz"
        if not processed.is_file():
            raise FileNotFoundError(
                f"Prepared FD001 data not found at {processed}. Run 'phm prepare' first."
            )
        if (self.artifact_dir / model / version).exists():
            raise FileExistsError(f"Model version already exists: {model}/{version}")
        if self.jobs.has_version_conflict(model, version):
            raise ValueError(f"A queued or completed candidate already targets {model}/{version}")
        job = self.jobs.enqueue(
            model,
            version,
            params,
            preset=preset,
            parameter_source=parameter_source,
            auto_start_worker=auto_start_worker,
            parameter_overrides={
                name: value for name, value in overrides.items() if value is not None
            },
        )
        worker: dict[str, Any]
        if auto_start_worker:
            try:
                worker = self.supervisor.ensure_worker()
            except RuntimeError as error:
                worker = {
                    "action": "start_failed",
                    "alive": False,
                    "error": str(error),
                }
        else:
            worker = {"action": "not_requested", **self.supervisor.status()}
        return self._decorate_job(job, worker)

    def get_training_job(self, job_id: str) -> dict[str, Any]:
        return self._decorate_job(self.jobs.require(job_id), self.supervisor.status())

    def list_training_jobs(self, limit: int = 20) -> dict[str, Any]:
        jobs = self.jobs.list(limit)
        return {
            "jobs": [
                {**job, "queue_position": self.jobs.queue_position(str(job["id"]))} for job in jobs
            ],
            "worker": self.supervisor.status(),
        }

    def cancel_training_job(self, job_id: str) -> dict[str, Any]:
        return self.jobs.request_cancel(job_id)

    def promote_candidate_model(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.require(job_id)
        if job["status"] != "succeeded":
            raise ValueError("Only a succeeded training job can be promoted")
        if not job["candidate_path"]:
            raise ValueError("Training job has no candidate artifact")
        registry = self.registry.promote(
            str(job["model"]),
            str(job["candidate_version"]),
            Path(str(job["candidate_path"])),
        )
        return {
            "promoted": {
                "model": job["model"],
                "version": job["candidate_version"],
                "job_id": job_id,
            },
            "registry": registry,
        }

    def rollback_model(self, model: str) -> dict[str, Any]:
        return self.registry.rollback(model)

    def get_model_registry(self) -> dict[str, Any]:
        return self.registry.snapshot()

    def worker_status(self) -> dict[str, Any]:
        return self.supervisor.status()

    def ensure_training_worker(self) -> dict[str, Any]:
        return self.supervisor.ensure_worker()

    def stop_training_worker(self) -> dict[str, Any]:
        return self.supervisor.request_stop()

    def _decorate_job(self, job: dict[str, Any], worker: dict[str, Any]) -> dict[str, Any]:
        return {
            **job,
            "queue_position": self.jobs.queue_position(str(job["id"])),
            "worker": worker,
            "published": self.registry.snapshot().get("active", {}).get(job["model"])
            == job["candidate_version"],
        }

    @staticmethod
    def _validate_model(model: str) -> None:
        if model not in MODEL_NAMES:
            raise ValueError("model must be 'lstm' or 'transformer'")

    @staticmethod
    def _validate_version(version: str) -> None:
        if version == "active" or not VERSION_PATTERN.fullmatch(version):
            raise ValueError("version must be 1-64 letters, digits, dots, underscores, or hyphens")

    @staticmethod
    def _validate_training_params(
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        seed: int,
        device: str,
        amp: bool,
    ) -> None:
        if not 1 <= epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        if not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be greater than 0 and at most 1")
        if not 1 <= patience <= epochs:
            raise ValueError("patience must be between 1 and epochs")
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("seed must be between 0 and 2147483647")
        if not DEVICE_PATTERN.fullmatch(device):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")
        if amp and device in {"cpu", "mps"}:
            raise ValueError("amp is currently supported only with CUDA or auto")
