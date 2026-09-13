from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import column_names
from .models import create_model


class InferenceService:
    def __init__(self, data_dir: Path, artifact_dir: Path) -> None:
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self._data: dict[str, np.ndarray] | None = None
        self._models: dict[tuple[str, str], tuple[torch.nn.Module, dict, dict, dict]] = {}

    def _prepared(self) -> dict[str, np.ndarray]:
        if self._data is None:
            path = self.data_dir / "processed" / "fd001.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Prepared FD001 data not found at {path}. Run 'phm prepare' first."
                )
            with np.load(path) as data:
                self._data = {key: data[key] for key in data.files}
        return self._data

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for name in ("lstm", "transformer"):
            root = self.artifact_dir / name
            if not root.is_dir():
                continue
            for version in sorted(root.iterdir()):
                if not version.is_dir():
                    continue
                config_path = version / "config.json"
                metrics_path = version / "metrics.json"
                model_path = version / "model.pt"
                if not (config_path.is_file() and metrics_path.is_file() and model_path.is_file()):
                    continue
                models.append(
                    {
                        "model": name,
                        "version": version.name,
                        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                    }
                )
        return models

    def _load_model(self, name: str, version: str):
        key = (name, version)
        if key in self._models:
            return self._models[key]
        target = self.artifact_dir / name / version
        config_path = target / "config.json"
        metrics_path = target / "metrics.json"
        interval_path = target / "residual_quantiles.json"
        if not (target / "model.pt").is_file():
            raise FileNotFoundError(f"Model artifact not found: {name}/{version}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        interval = json.loads(interval_path.read_text(encoding="utf-8"))
        model = create_model(
            name,
            int(config["input_size"]),
            int(config["window_size"]),
            config["model_config"],
        )
        state = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._models[key] = (model, config, metrics, interval)
        return self._models[key]

    def _unit_index(self, unit_id: int) -> int:
        ids = self._prepared()["test_unit_ids"].astype(np.int64)
        matches = np.flatnonzero(ids == unit_id)
        if len(matches) != 1:
            raise ValueError(f"Unknown FD001 test engine: {unit_id}")
        return int(matches[0])

    def inspect_engine(self, unit_id: int) -> dict[str, Any]:
        data = self._prepared()
        index = self._unit_index(unit_id)
        window = data["test_x"][index]
        finite = bool(np.isfinite(window).all())
        return {
            "dataset": "FD001",
            "split": "test",
            "unit_id": unit_id,
            "last_cycle": int(data["test_last_cycles"][index]),
            "window_size": int(data["window_size"]),
            "feature_count": int(window.shape[1]),
            "missing_values": int(np.isnan(window).sum()),
            "finite": finite,
            "data_quality": "passed" if finite else "failed",
        }

    def predict_rul(self, unit_id: int, model_name: str, version: str = "v1") -> dict[str, Any]:
        quality = self.inspect_engine(unit_id)
        if quality["data_quality"] != "passed":
            raise ValueError("Engine window failed data-quality checks")
        data = self._prepared()
        index = self._unit_index(unit_id)
        model, config, metrics, interval = self._load_model(model_name, version)
        values = torch.from_numpy(data["test_x"][index : index + 1])
        with torch.inference_mode():
            prediction = max(0.0, float(model(values).item()))
        lower = max(0.0, prediction + float(interval["lower_delta"]))
        upper = max(lower, prediction + float(interval["upper_delta"]))
        return {
            **quality,
            "model": model_name,
            "version": version,
            "predicted_rul": round(prediction, 3),
            "interval": {
                "coverage": interval["coverage"],
                "low": round(lower, 3),
                "high": round(upper, 3),
                "method": "validation residual quantiles",
            },
            "validation_metrics": metrics["validation"],
            "test_metrics_reference": metrics["test"],
            "model_git_sha": config.get("git_sha", ""),
        }

    def compare_models(self, unit_id: int, version: str = "v1") -> dict[str, Any]:
        predictions = [self.predict_rul(unit_id, name, version) for name in ("lstm", "transformer")]
        recommended = min(predictions, key=lambda item: item["validation_metrics"]["rmse"])
        return {
            "dataset": "FD001",
            "unit_id": unit_id,
            "predictions": predictions,
            "absolute_difference": round(
                abs(predictions[0]["predicted_rul"] - predictions[1]["predicted_rul"]),
                3,
            ),
            "recommended_model": recommended["model"],
            "selection_rule": "lowest validation RMSE",
        }

    def degradation_evidence(self, unit_id: int, top_k: int = 5) -> dict[str, Any]:
        data = self._prepared()
        index = self._unit_index(unit_id)
        window = data["test_x"][index]
        x = np.arange(window.shape[0], dtype=np.float32)
        centered_x = x - x.mean()
        denominator = float(np.square(centered_x).sum())
        slopes = (centered_x[:, None] * (window - window.mean(axis=0))).sum(axis=0) / denominator
        shifts = window[-1] - window[:10].mean(axis=0)
        ranking = np.argsort(np.abs(shifts))[::-1][: max(1, min(top_k, window.shape[1]))]
        selected = data["selected_feature_indices"].astype(np.int64)
        names = column_names()
        evidence = []
        for feature_index in ranking:
            raw_index = int(selected[feature_index]) + 2
            evidence.append(
                {
                    "feature": names[raw_index],
                    "normalized_shift": round(float(shifts[feature_index]), 4),
                    "normalized_slope_per_cycle": round(float(slopes[feature_index]), 5),
                    "last_normalized_value": round(float(window[-1, feature_index]), 4),
                }
            )
        return {
            "dataset": "FD001",
            "unit_id": unit_id,
            "window_size": int(window.shape[0]),
            "evidence": evidence,
            "caveat": ("Trend evidence is descriptive and does not establish physical causality."),
        }
