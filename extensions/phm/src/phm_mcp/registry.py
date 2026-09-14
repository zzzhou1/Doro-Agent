from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .jobs import VERSION_PATTERN

MODEL_NAMES = ("lstm", "transformer")
REQUIRED_ARTIFACTS = (
    "model.pt",
    "config.json",
    "metrics.json",
    "history.json",
    "residual_quantiles.json",
)


class ModelRegistry:
    """Atomic active-version registry with explicit promotion and rollback."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.path = artifact_dir / "registry.json"

    def _default(self) -> dict[str, Any]:
        active = {
            model: "v1"
            for model in MODEL_NAMES
            if (self.artifact_dir / model / "v1" / "model.pt").is_file()
        }
        return {
            "schema_version": 1,
            "active": active,
            "history": {model: [] for model in MODEL_NAMES},
            "updated_at": None,
        }

    def snapshot(self) -> dict[str, Any]:
        if not self.path.is_file():
            registry = self._default()
        else:
            registry = json.loads(self.path.read_text(encoding="utf-8"))
        registry["available"] = {model: self.available_versions(model) for model in MODEL_NAMES}
        return registry

    def available_versions(self, model: str) -> list[str]:
        self._validate_model(model)
        root = self.artifact_dir / model
        if not root.is_dir():
            return []
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and all((path / name).is_file() for name in REQUIRED_ARTIFACTS)
        )

    def resolve(self, model: str, version: str) -> str:
        self._validate_model(model)
        if version != "active":
            self._validate_version(version)
            return version
        active = self.snapshot()["active"].get(model)
        if not active:
            raise FileNotFoundError(f"No active model is registered for {model}")
        return str(active)

    def promote(self, model: str, version: str, candidate_path: Path) -> dict[str, Any]:
        self._validate_model(model)
        self._validate_version(version)
        candidate = candidate_path.resolve()
        candidates_root = (self.artifact_dir / "candidates").resolve()
        if candidates_root not in candidate.parents:
            raise ValueError("Candidate path is outside the managed candidates directory")
        self._validate_artifact(candidate, model, version)

        target_root = self.artifact_dir / model
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / version
        if target.exists():
            raise FileExistsError(f"Model version already exists: {model}/{version}")
        staging = target_root / f".{version}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copytree(candidate, staging)
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        registry = self.snapshot()
        registry.pop("available", None)
        previous = registry["active"].get(model)
        history = registry["history"].setdefault(model, [])
        if previous and previous != version:
            history.append(previous)
        registry["active"][model] = version
        registry["updated_at"] = datetime.now(UTC).isoformat()
        self._write(registry)
        return self.snapshot()

    def rollback(self, model: str) -> dict[str, Any]:
        self._validate_model(model)
        registry = self.snapshot()
        registry.pop("available", None)
        history = registry["history"].setdefault(model, [])
        if not history:
            raise ValueError(f"No previous active version is available for {model}")
        previous = str(history.pop())
        if previous not in self.available_versions(model):
            raise FileNotFoundError(f"Rollback artifact is missing: {model}/{previous}")
        registry["active"][model] = previous
        registry["updated_at"] = datetime.now(UTC).isoformat()
        self._write(registry)
        return self.snapshot()

    def _write(self, registry: dict[str, Any]) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(registry, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_model(model: str) -> None:
        if model not in MODEL_NAMES:
            raise ValueError("model must be 'lstm' or 'transformer'")

    @staticmethod
    def _validate_version(version: str) -> None:
        if not VERSION_PATTERN.fullmatch(version) or version == "active":
            raise ValueError("version must be 1-64 letters, digits, dots, underscores, or hyphens")

    @staticmethod
    def _validate_artifact(candidate: Path, model: str, version: str) -> None:
        missing = [name for name in REQUIRED_ARTIFACTS if not (candidate / name).is_file()]
        if missing:
            raise ValueError(f"Candidate is missing artifacts: {', '.join(missing)}")
        config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
        if config.get("model") != model or config.get("version") != version:
            raise ValueError("Candidate model/version does not match the promotion request")
