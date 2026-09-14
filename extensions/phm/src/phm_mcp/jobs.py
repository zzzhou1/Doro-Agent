from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FINAL_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
VERSION_BLOCKING_STATUSES = ACTIVE_STATUSES | {"succeeded"}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    """Small, process-safe SQLite queue for local PHM training jobs."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    candidate_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    progress_json TEXT,
                    metrics_json TEXT,
                    candidate_path TEXT,
                    error TEXT,
                    worker_pid INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_jobs_status_created
                ON training_jobs(status, created_at)
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["params"] = json.loads(result.pop("params_json"))
        progress = result.pop("progress_json")
        metrics = result.pop("metrics_json")
        result["progress"] = json.loads(progress) if progress else None
        result["metrics"] = json.loads(metrics) if metrics else None
        return result

    def enqueue(self, model: str, candidate_version: str, params: dict[str, Any]) -> dict:
        job_id = f"train-{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_jobs (
                    id, model, candidate_version, status, params_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, model, candidate_version, json.dumps(params), now, now),
            )
        return self.require(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM training_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._decode(row)

    def require(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"Unknown training job: {job_id}")
        return job

    def list(self, limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode(row) for row in rows]  # type: ignore[misc]

    def has_version_conflict(self, model: str, candidate_version: str) -> bool:
        placeholders = ",".join("?" for _ in VERSION_BLOCKING_STATUSES)
        values = [model, candidate_version, *sorted(VERSION_BLOCKING_STATUSES)]
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT 1 FROM training_jobs
                WHERE model = ? AND candidate_version = ?
                  AND status IN ({placeholders})
                LIMIT 1
                """,
                values,
            ).fetchone()
        return row is not None

    def claim_next(self, worker_pid: int) -> dict | None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id FROM training_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            job_id = str(row["id"])
            connection.execute(
                """
                UPDATE training_jobs
                SET status = 'running', worker_pid = ?, started_at = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (worker_pid, now, now, job_id),
            )
            connection.commit()
        return self.require(job_id)

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE training_jobs SET progress_json = ?, updated_at = ?
                WHERE id = ? AND status IN ('running', 'cancelling')
                """,
                (json.dumps(progress), utc_now(), job_id),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"Training job is not running: {job_id}")

    def request_cancel(self, job_id: str) -> dict:
        now = utc_now()
        with self._connect() as connection:
            queued = connection.execute(
                """
                UPDATE training_jobs
                SET status = 'cancelled', finished_at = ?, updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, job_id),
            )
            if queued.rowcount == 0:
                running = connection.execute(
                    """
                    UPDATE training_jobs
                    SET status = 'cancelling', updated_at = ?
                    WHERE id = ? AND status = 'running'
                    """,
                    (now, job_id),
                )
            else:
                running = None
        if queued.rowcount == 1 or (running is not None and running.rowcount == 1):
            return self.require(job_id)
        job = self.require(job_id)
        if job["status"] == "cancelling":
            return job
        raise ValueError(f"Cannot cancel job in status '{job['status']}'")

    def cancellation_requested(self, job_id: str) -> bool:
        return self.require(job_id)["status"] == "cancelling"

    def mark_succeeded(self, job_id: str, candidate_path: Path, metrics: dict[str, Any]) -> dict:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE training_jobs
                SET status = 'succeeded', candidate_path = ?, metrics_json = ?,
                    finished_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    str(candidate_path.resolve()),
                    json.dumps(metrics),
                    now,
                    now,
                    job_id,
                ),
            )
        if cursor.rowcount != 1:
            job = self.require(job_id)
            raise ValueError(f"Cannot complete job in status '{job['status']}'")
        return self.require(job_id)

    def mark_failed(self, job_id: str, error: str) -> dict:
        return self._finish(job_id, "failed", error=error[:4000])

    def mark_cancelled(self, job_id: str) -> dict:
        return self._finish(job_id, "cancelled")

    def _finish(self, job_id: str, status: str, **fields: Any) -> dict:
        if status not in FINAL_STATUSES:
            raise ValueError(f"Invalid final status: {status}")
        assignments = ["status = ?", "finished_at = ?", "updated_at = ?"]
        values: list[Any] = [status, utc_now(), utc_now()]
        for name, value in fields.items():
            if name not in {"candidate_path", "metrics_json", "error"}:
                raise ValueError(f"Invalid job field: {name}")
            assignments.append(f"{name} = ?")
            values.append(value)
        values.append(job_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE training_jobs SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        if cursor.rowcount != 1:
            raise ValueError(f"Unknown training job: {job_id}")
        return self.require(job_id)

    def fail_interrupted(self) -> int:
        """Fail jobs left running after a single-worker process restart."""
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE training_jobs
                SET status = 'failed',
                    error = 'Training worker stopped before the job completed',
                    finished_at = ?, updated_at = ?
                WHERE status IN ('running', 'cancelling')
                """,
                (now, now),
            )
        return cursor.rowcount
