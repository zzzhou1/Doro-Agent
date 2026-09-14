from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .jobs import JobStore


class WorkerSupervisor:
    """Start and control the single local training worker."""

    def __init__(self, data_dir: Path, artifact_dir: Path, runtime_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.runtime_dir = runtime_dir.resolve()
        self.store = JobStore(self.runtime_dir / "jobs.sqlite")

    def ensure_worker(self, idle_timeout: float = 300.0) -> dict[str, Any]:
        if idle_timeout < 30:
            raise ValueError("idle_timeout must be at least 30 seconds")
        current = self.store.worker_status()
        if current["alive"]:
            return {"action": "already_running", **current}

        worker_id = uuid.uuid4().hex
        reservation = self.store.reserve_worker_start(worker_id)
        if not reservation["reserved"]:
            return {"action": "already_running", **reservation["worker"]}

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.runtime_dir / "worker.stdout.log"
        stderr_path = self.runtime_dir / "worker.stderr.log"
        command = [
            sys.executable,
            "-m",
            "phm_mcp.worker",
            "--data-dir",
            str(self.data_dir),
            "--artifact-dir",
            str(self.artifact_dir),
            "--runtime-dir",
            str(self.runtime_dir),
            "--worker-id",
            worker_id,
            "--idle-timeout",
            str(idle_timeout),
        ]
        creationflags = 0
        popen_options: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NO_WINDOW
            )
            popen_options = {"creationflags": creationflags}

        try:
            with (
                stdout_path.open("a", encoding="utf-8") as stdout,
                stderr_path.open("a", encoding="utf-8") as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=str(self.data_dir.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    close_fds=True,
                    env=os.environ.copy(),
                    **popen_options,
                )
            self.store.record_worker_process(worker_id, process.pid)
        except Exception as error:
            self.store.release_worker(worker_id, "failed", str(error))
            raise RuntimeError(f"Failed to start PHM training worker: {error}") from error

        return {
            "action": "started",
            **self.store.worker_status(),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    def status(self) -> dict[str, Any]:
        return self.store.worker_status()

    def request_stop(self) -> dict[str, Any]:
        return self.store.request_worker_stop()
