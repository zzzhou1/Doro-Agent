from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .jobs import ACTIVE_STATUSES, VERSION_PATTERN, JobStore
from .registry import MODEL_NAMES, ModelRegistry

DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda(?::[0-9]+)?|mps)$")


class AdminService:
    """Validated control plane for queued training and model promotion."""

    def __init__(self, data_dir: Path, artifact_dir: Path, runtime_dir: Path) -> None:
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.runtime_dir = runtime_dir
        self.jobs = JobStore(runtime_dir / "jobs.sqlite")
        self.registry = ModelRegistry(artifact_dir)

    def submit_training_job(
        self,
        *,
        model: str,
        version: str | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 8,
        seed: int = 42,
        device: str = "auto",
        amp: bool = False,
    ) -> dict[str, Any]:
        self._validate_model(model)
        version = version or f"candidate-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        self._validate_version(version)
        self._validate_training_params(
            epochs, batch_size, learning_rate, patience, seed, device, amp
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
        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
            "device": device,
            "amp": amp,
        }
        return self.jobs.enqueue(model, version, params)

    def get_training_job(self, job_id: str) -> dict[str, Any]:
        return self.jobs.require(job_id)

    def list_training_jobs(self, limit: int = 20) -> dict[str, Any]:
        return {"jobs": self.jobs.list(limit)}

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
        jobs = self.jobs.list(200)
        active = [job for job in jobs if job["status"] in ACTIVE_STATUSES]
        return {
            "active_jobs": active,
            "worker_required": bool(active),
            "hint": "Run 'uv run --project extensions/phm phm-worker' to process queued jobs.",
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
