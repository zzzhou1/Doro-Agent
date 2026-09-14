from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_RUNTIME_DIR
from .jobs import JobStore
from .training import TrainingCancelled, train_model


class TrainingWorker:
    """Single local worker that executes queued jobs outside MCP requests."""

    def __init__(self, data_dir: Path, artifact_dir: Path, runtime_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.store = JobStore(runtime_dir.resolve() / "jobs.sqlite")

    def run_once(self) -> dict[str, Any] | None:
        job = self.store.claim_next(os.getpid())
        if job is None:
            return None
        job_id = str(job["id"])
        candidate_path = self.artifact_dir / "candidates" / job_id

        def progress_callback(progress: dict[str, Any]) -> None:
            self.store.update_progress(job_id, progress)

        def cancel_check() -> bool:
            return self.store.cancellation_requested(job_id)

        try:
            params = dict(job["params"])
            result = train_model(
                str(job["model"]),
                self.data_dir / "processed" / "fd001.npz",
                self.artifact_dir,
                version=str(job["candidate_version"]),
                target_dir=candidate_path,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                **params,
            )
            if cancel_check():
                return self.store.mark_cancelled(job_id)
            return self.store.mark_succeeded(job_id, candidate_path, result["metrics"])
        except TrainingCancelled:
            return self.store.mark_cancelled(job_id)
        except Exception as error:
            if cancel_check():
                return self.store.mark_cancelled(job_id)
            details = "".join(traceback.format_exception_only(type(error), error)).strip()
            return self.store.mark_failed(job_id, details)

    def run_forever(self, poll_interval: float = 1.0) -> None:
        if poll_interval < 0.1:
            raise ValueError("poll_interval must be at least 0.1 seconds")
        interrupted = self.store.fail_interrupted()
        if interrupted:
            print(f"Marked {interrupted} interrupted training job(s) as failed.", flush=True)
        print(f"PHM training worker started (pid={os.getpid()}).", flush=True)
        try:
            while True:
                job = self.run_once()
                if job is None:
                    time.sleep(poll_interval)
                else:
                    print(f"Training job {job['id']} -> {job['status']}", flush=True)
        except KeyboardInterrupt:
            print("PHM training worker stopped.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process queued PHM training jobs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    worker = TrainingWorker(args.data_dir, args.artifact_dir, args.runtime_dir)
    if args.once:
        job = worker.run_once()
        print("No queued jobs." if job is None else f"{job['id']}: {job['status']}")
    else:
        worker.run_forever(args.poll_interval)


if __name__ == "__main__":
    main()
