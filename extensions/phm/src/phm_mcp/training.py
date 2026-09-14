from __future__ import annotations

import json
import random
import subprocess
import time
from collections.abc import Callable
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


class TrainingCancelled(RuntimeError):
    """Raised when an asynchronous job requests cooperative cancellation."""


def resolve_device(requested: str | None = "auto") -> torch.device:
    name = requested or "auto"
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if name == "cuda" or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this PyTorch build or machine has no available CUDA device"
            )
        device = torch.device(name)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {device.index}")
        return device
    raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")


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
    model: nn.Module, values: np.ndarray, batch_size: int, device: str | torch.device
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
    device: str | None = "auto",
    amp: bool = False,
    target_dir: Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    if model_name not in {"lstm", "transformer"}:
        raise ValueError("model must be 'lstm' or 'transformer'")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not processed_path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {processed_path}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = resolve_device(device)
    use_amp = bool(amp and resolved_device.type == "cuda")
    if amp and not use_amp:
        raise RuntimeError("AMP is supported only when training on CUDA")
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    if cancelled():
        raise TrainingCancelled("Training was cancelled before it started")

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
        pin_memory=resolved_device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if cancelled():
            raise TrainingCancelled(f"Training was cancelled before epoch {epoch}")
        model.train()
        losses: list[float] = []
        for batch_index, (batch_x, batch_y) in enumerate(loader, start=1):
            if batch_index % 10 == 0 and cancelled():
                raise TrainingCancelled(f"Training was cancelled during epoch {epoch}")
            batch_x = batch_x.to(resolved_device, non_blocking=use_amp)
            batch_y = batch_y.to(resolved_device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = loss_fn(model(batch_x), batch_y)
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = predict_batches(model, validation_x, batch_size, resolved_device)
        validation_loss = float(np.mean(np.square(validation_prediction - validation_y)))
        train_loss = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        should_stop = False
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                should_stop = True
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "epochs_requested": epochs,
                    "train_mse": train_loss,
                    "validation_mse": validation_loss,
                    "best_validation_mse": best_loss,
                    "stale_epochs": stale_epochs,
                    "device": str(resolved_device),
                }
            )
        if should_stop:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    if cancelled():
        raise TrainingCancelled("Training was cancelled before evaluation")
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

    target = target_dir.resolve() if target_dir else artifact_dir / model_name / version
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
            "device": str(resolved_device),
            "amp": use_amp,
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
