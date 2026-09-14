from __future__ import annotations

import argparse
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_RUNTIME_DIR
from .jobs import JobStore
from .training import TrainingCancelled, train_model


class TrainingWorker:
    """Single local worker that executes queued jobs outside MCP requests."""

    def __init__(
        self,
        data_dir: Path,
        artifact_dir: Path,
        runtime_dir: Path,
        worker_id: str | None = None,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.store = JobStore(runtime_dir.resolve() / "jobs.sqlite")
        self.worker_id = worker_id
        self._status = "idle"
        self._current_job_id: str | None = None
        self._heartbeat_stop = threading.Event()

    def run_once(self) -> dict[str, Any] | None:
        job = self.store.claim_next(os.getpid())
        if job is None:
            return None
        job_id = str(job["id"])
        candidate_path = self.artifact_dir / "candidates" / job_id
        self._status = "busy"
        self._current_job_id = job_id
        self._heartbeat()

        def progress_callback(progress: dict[str, Any]) -> None:
            self.store.update_progress(job_id, progress)
            self._heartbeat()

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
        finally:
            self._status = "idle"
            self._current_job_id = None
            self._heartbeat()

    def _heartbeat(self) -> bool:
        if self.worker_id is None:
            return True
        return self.store.heartbeat_worker(
            self.worker_id,
            os.getpid(),
            self._status,
            self._current_job_id,
        )

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._heartbeat_stop.wait(interval):
            try:
                if not self._heartbeat():
                    self._heartbeat_stop.set()
            except Exception:
                # A transient SQLite lock should not kill an active training job.
                continue

    def run_forever(
        self,
        poll_interval: float = 1.0,
        idle_timeout: float = 300.0,
        heartbeat_interval: float = 5.0,
    ) -> None:
        if poll_interval < 0.1:
            raise ValueError("poll_interval must be at least 0.1 seconds")
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be non-negative")
        if heartbeat_interval < 1:
            raise ValueError("heartbeat_interval must be at least 1 second")
        self.worker_id = self.worker_id or uuid.uuid4().hex
        reservation = self.store.reserve_worker_start(self.worker_id)
        owner = reservation["worker"].get("worker_id") == self.worker_id
        if not reservation["reserved"] and not owner:
            print(
                f"PHM training worker already running (pid={reservation['worker'].get('pid')}).",
                flush=True,
            )
            return
        if not self.store.activate_worker(self.worker_id, os.getpid()):
            print("PHM training worker could not acquire its lease.", flush=True)
            return
        interrupted = self.store.fail_interrupted()
        if interrupted:
            print(f"Marked {interrupted} interrupted training job(s) as failed.", flush=True)
        print(f"PHM training worker started (pid={os.getpid()}).", flush=True)
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_interval,),
            name="phm-worker-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        idle_since = time.monotonic()
        try:
            while True:
                if self.store.worker_stop_requested(self.worker_id):
                    print("PHM training worker stop requested.", flush=True)
                    break
                job = self.run_once()
                if job is None:
                    if idle_timeout and time.monotonic() - idle_since >= idle_timeout:
                        print("PHM training worker idle timeout reached.", flush=True)
                        break
                    time.sleep(poll_interval)
                else:
                    idle_since = time.monotonic()
                    print(f"Training job {job['id']} -> {job['status']}", flush=True)
        except KeyboardInterrupt:
            print("PHM training worker stopped.", flush=True)
        finally:
            self._heartbeat_stop.set()
            heartbeat_thread.join(timeout=heartbeat_interval + 1)
            self.store.release_worker(self.worker_id, "stopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process queued PHM training jobs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--idle-timeout", type=float, default=300.0)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--worker-id")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    worker = TrainingWorker(
        args.data_dir,
        args.artifact_dir,
        args.runtime_dir,
        worker_id=args.worker_id,
    )
    if args.once:
        job = worker.run_once()
        print("No queued jobs." if job is None else f"{job['id']}: {job['status']}")
    else:
        worker.run_forever(args.poll_interval, args.idle_timeout, args.heartbeat_interval)


if __name__ == "__main__":
    main()
