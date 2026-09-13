from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import create_model


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    nasa_score: float


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Metrics:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    return Metrics(
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mae=float(np.mean(np.abs(error))),
        nasa_score=float(
            np.sum(
                np.where(
                    error < 0,
                    np.exp(-error / 13.0) - 1,
                    np.exp(error / 10.0) - 1,
                )
            )
        ),
    )


def predict_batches(
    model: nn.Module, values: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            output = torch.clamp(model(batch), min=0.0)
            chunks.append(output.cpu().numpy())
    return np.concatenate(chunks)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def train_model(
    model_name: str,
    processed_path: Path,
    artifact_dir: Path,
    *,
    version: str = "v1",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 8,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    if model_name not in {"lstm", "transformer"}:
        raise ValueError("model must be 'lstm' or 'transformer'")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(processed_path) as data:
        train_x = data["train_x"]
        train_y = data["train_y"]
        validation_x = data["validation_x"]
        validation_y = data["validation_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]
        window_size = int(data["window_size"])

    model_config = (
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}
        if model_name == "lstm"
        else {
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
        }
    )
    model = create_model(model_name, train_x.shape[2], window_size, model_config)
    model.to(resolved_device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = predict_batches(model, validation_x, batch_size, resolved_device)
        validation_loss = float(np.mean(np.square(validation_prediction - validation_y)))
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    model.to(resolved_device)
    validation_prediction = predict_batches(model, validation_x, batch_size, resolved_device)
    test_prediction = predict_batches(model, test_x, batch_size, resolved_device)
    validation_metrics = calculate_metrics(validation_y, validation_prediction)
    test_metrics = calculate_metrics(test_y, test_prediction)
    residuals = validation_y - validation_prediction
    interval = {
        "lower_delta": float(np.quantile(residuals, 0.1)),
        "upper_delta": float(np.quantile(residuals, 0.9)),
        "coverage": 0.8,
    }

    target = artifact_dir / model_name / version
    target.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, target / "model.pt")
    config = {
        "model": model_name,
        "version": version,
        "input_size": int(train_x.shape[2]),
        "window_size": window_size,
        "model_config": model_config,
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "seed": seed,
            "device": resolved_device,
        },
        "git_sha": _git_sha(),
    }
    metrics = {
        "validation": asdict(validation_metrics),
        "test": asdict(test_metrics),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - started,
    }
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (target / "residual_quantiles.json").write_text(
        json.dumps(interval, indent=2), encoding="utf-8"
    )
    (target / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "artifact": str(target),
        "config": config,
        "metrics": metrics,
        "interval": interval,
    }
