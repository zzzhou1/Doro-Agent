from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR
from .data import download_fd001, prepare_fd001
from .inference import InferenceService
from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NASA C-MAPSS PHM extension")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download official NASA FD001 files")
    prepare = commands.add_parser("prepare", help="Build leakage-safe windows")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    train = commands.add_parser("train", help="Train one RUL model")
    train.add_argument("--model", choices=["lstm", "transformer"], required=True)
    train.add_argument("--version", default="v1")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    train.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    evaluate = commands.add_parser("evaluate", help="Show saved metrics")
    evaluate.add_argument("--model", choices=["lstm", "transformer"])
    predict = commands.add_parser("predict", help="Predict one test engine")
    predict.add_argument("--unit-id", type=int, required=True)
    predict.add_argument("--model", choices=["lstm", "transformer"])
    predict.add_argument("--version", default="active")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    if args.command == "download":
        result = [str(path) for path in download_fd001(data_dir)]
    elif args.command == "prepare":
        result = asdict(
            prepare_fd001(
                data_dir,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        )
        result["output_path"] = str(result["output_path"])
    elif args.command == "train":
        result = train_model(
            args.model,
            data_dir / "processed" / "fd001.npz",
            artifact_dir,
            version=args.version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            amp=args.amp,
        )
    else:
        service = InferenceService(data_dir, artifact_dir)
        if args.command == "evaluate":
            result = {"models": service.list_models()}
            if args.model:
                result["models"] = [
                    model for model in result["models"] if model["model"] == args.model
                ]
        elif args.model:
            result = service.predict_rul(args.unit_id, args.model, args.version)
        else:
            result = service.compare_models(args.unit_id, args.version)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
