from __future__ import annotations

import argparse
import json
from pathlib import Path

from .admin import AdminService
from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_RUNTIME_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage asynchronous PHM training")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    submit = commands.add_parser("submit", help="Queue a training job")
    submit.add_argument("--model", choices=["lstm", "transformer"], required=True)
    submit.add_argument("--version")
    submit.add_argument("--preset", choices=["smoke", "standard", "thorough"], default="standard")
    submit.add_argument("--epochs", type=int)
    submit.add_argument("--batch-size", type=int)
    submit.add_argument("--learning-rate", type=float)
    submit.add_argument("--patience", type=int)
    submit.add_argument("--seed", type=int)
    submit.add_argument("--device")
    submit.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    submit.add_argument(
        "--auto-start-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    submit.add_argument(
        "--confirm-preset",
        action="store_true",
        help="Confirm use of the selected preset when no hyperparameters are provided",
    )

    status = commands.add_parser("status", help="Read one job")
    status.add_argument("job_id")
    listing = commands.add_parser("list", help="List recent jobs")
    listing.add_argument("--limit", type=int, default=20)
    cancel = commands.add_parser("cancel", help="Cancel one job")
    cancel.add_argument("job_id")
    promote = commands.add_parser("promote", help="Promote one succeeded candidate")
    promote.add_argument("job_id")
    rollback = commands.add_parser("rollback", help="Roll back an active model")
    rollback.add_argument("--model", choices=["lstm", "transformer"], required=True)
    commands.add_parser("registry", help="Show active and available models")
    commands.add_parser("worker-status", help="Show managed worker heartbeat and queue")
    commands.add_parser("worker-start", help="Start or reuse the managed worker")
    commands.add_parser("worker-stop", help="Request the managed worker to stop")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    service = AdminService(
        args.data_dir.resolve(),
        args.artifact_dir.resolve(),
        args.runtime_dir.resolve(),
    )
    if args.command == "submit":
        result = service.submit_training_job(
            model=args.model,
            version=args.version,
            preset=args.preset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
            amp=args.amp,
            auto_start_worker=args.auto_start_worker,
            preset_confirmed=args.confirm_preset,
        )
    elif args.command == "status":
        result = service.get_training_job(args.job_id)
    elif args.command == "list":
        result = service.list_training_jobs(args.limit)
    elif args.command == "cancel":
        result = service.cancel_training_job(args.job_id)
    elif args.command == "promote":
        result = service.promote_candidate_model(args.job_id)
    elif args.command == "rollback":
        result = service.rollback_model(args.model)
    elif args.command == "worker-status":
        result = service.worker_status()
    elif args.command == "worker-start":
        result = service.ensure_training_worker()
    elif args.command == "worker-stop":
        result = service.stop_training_worker()
    else:
        result = service.get_model_registry()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
