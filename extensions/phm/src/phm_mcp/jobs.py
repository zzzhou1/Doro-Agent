from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

FINAL_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
VERSION_BLOCKING_STATUSES = ACTIVE_STATUSES | {"succeeded"}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
WORKER_ACTIVE_STATUSES = {"starting", "idle", "busy", "stopping"}
WORKER_LEASE_SECONDS = 30


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def utc_after(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


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
                CREATE TABLE IF NOT EXISTS worker_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    worker_id TEXT NOT NULL,
                    pid INTEGER,
                    status TEXT NOT NULL,
                    current_job_id TEXT,
                    started_at TEXT NOT NULL,
                    last_heartbeat_at TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(training_jobs)").fetchall()
            }
            migrations = {
                "preset": "TEXT NOT NULL DEFAULT 'legacy'",
                "parameter_source": "TEXT NOT NULL DEFAULT 'legacy'",
                "auto_start_worker": "INTEGER NOT NULL DEFAULT 0",
                "overrides_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE training_jobs ADD COLUMN {name} {declaration}")
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
        result["auto_start_worker"] = bool(result["auto_start_worker"])
        result["parameter_overrides"] = json.loads(result.pop("overrides_json"))
        return result

    def enqueue(
        self,
        model: str,
        candidate_version: str,
        params: dict[str, Any],
        *,
        preset: str = "legacy",
        parameter_source: str = "legacy",
        auto_start_worker: bool = False,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> dict:
        job_id = f"train-{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_jobs (
                    id, model, candidate_version, status, params_json,
                    preset, parameter_source, auto_start_worker, overrides_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    model,
                    candidate_version,
                    json.dumps(params),
                    preset,
                    parameter_source,
                    int(auto_start_worker),
                    json.dumps(parameter_overrides or {}),
                    now,
                    now,
                ),
            )
        return self.require(job_id)

    def queue_position(self, job_id: str) -> int | None:
        job = self.require(job_id)
        if job["status"] != "queued":
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS position FROM training_jobs
                WHERE status = 'queued' AND created_at <= ?
                """,
                (job["created_at"],),
            ).fetchone()
        return int(row["position"])

    def queue_depth(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS depth FROM training_jobs WHERE status = 'queued'"
            ).fetchone()
        return int(row["depth"])

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

    @staticmethod
    def _worker_payload(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {
                "worker_id": None,
                "pid": None,
                "status": "stopped",
                "alive": False,
                "current_job_id": None,
                "started_at": None,
                "last_heartbeat_at": None,
                "heartbeat_age_seconds": None,
                "lease_expires_at": None,
                "stop_requested": False,
                "error": None,
            }
        payload = dict(row)
        now = datetime.now(UTC)
        heartbeat = datetime.fromisoformat(str(payload["last_heartbeat_at"]))
        lease = datetime.fromisoformat(str(payload["lease_expires_at"]))
        alive = payload["status"] in WORKER_ACTIVE_STATUSES and lease > now
        if not alive and payload["status"] in WORKER_ACTIVE_STATUSES:
            payload["status"] = "stale"
        payload["alive"] = alive
        payload["stop_requested"] = bool(payload["stop_requested"])
        payload["heartbeat_age_seconds"] = max(0.0, (now - heartbeat).total_seconds())
        payload.pop("singleton", None)
        return payload

    def worker_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM worker_state WHERE singleton = 1").fetchone()
        return {**self._worker_payload(row), "queue_depth": self.queue_depth()}

    def reserve_worker_start(self, worker_id: str) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM worker_state WHERE singleton = 1").fetchone()
            current = self._worker_payload(row)
            if current["alive"]:
                connection.commit()
                return {"reserved": False, "worker": current}
            connection.execute(
                """
                INSERT INTO worker_state (
                    singleton, worker_id, pid, status, current_job_id,
                    started_at, last_heartbeat_at, lease_expires_at,
                    stop_requested, error
                ) VALUES (1, ?, NULL, 'starting', NULL, ?, ?, ?, 0, NULL)
                ON CONFLICT(singleton) DO UPDATE SET
                    worker_id = excluded.worker_id,
                    pid = NULL,
                    status = 'starting',
                    current_job_id = NULL,
                    started_at = excluded.started_at,
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    lease_expires_at = excluded.lease_expires_at,
                    stop_requested = 0,
                    error = NULL
                """,
                (worker_id, now, now, utc_after(WORKER_LEASE_SECONDS)),
            )
            connection.commit()
        return {"reserved": True, "worker": self.worker_status()}

    def record_worker_process(self, worker_id: str, pid: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE worker_state SET pid = ?, last_heartbeat_at = ?, lease_expires_at = ?
                WHERE singleton = 1 AND worker_id = ? AND status = 'starting'
                """,
                (pid, utc_now(), utc_after(WORKER_LEASE_SECONDS), worker_id),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("Worker start reservation was lost")

    def activate_worker(self, worker_id: str, pid: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE worker_state
                SET pid = ?, status = 'idle', last_heartbeat_at = ?,
                    lease_expires_at = ?, error = NULL
                WHERE singleton = 1 AND worker_id = ? AND status = 'starting'
                """,
                (pid, utc_now(), utc_after(WORKER_LEASE_SECONDS), worker_id),
            )
        return cursor.rowcount == 1

    def heartbeat_worker(
        self,
        worker_id: str,
        pid: int,
        status: str,
        current_job_id: str | None,
    ) -> bool:
        if status not in {"idle", "busy", "stopping"}:
            raise ValueError(f"Invalid worker status: {status}")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE worker_state
                SET pid = ?, status = ?, current_job_id = ?,
                    last_heartbeat_at = ?, lease_expires_at = ?
                WHERE singleton = 1 AND worker_id = ?
                """,
                (
                    pid,
                    status,
                    current_job_id,
                    utc_now(),
                    utc_after(WORKER_LEASE_SECONDS),
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def worker_stop_requested(self, worker_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT stop_requested FROM worker_state
                WHERE singleton = 1 AND worker_id = ?
                """,
                (worker_id,),
            ).fetchone()
        return bool(row and row["stop_requested"])

    def request_worker_stop(self) -> dict[str, Any]:
        current = self.worker_status()
        if not current["alive"]:
            return {"action": "already_stopped", **current}
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worker_state SET stop_requested = 1,
                    status = CASE WHEN status = 'idle' THEN 'stopping' ELSE status END
                WHERE singleton = 1
                """
            )
        return {"action": "stop_requested", **self.worker_status()}

    def release_worker(self, worker_id: str, status: str, error: str | None = None) -> None:
        if status not in {"stopped", "failed"}:
            raise ValueError(f"Invalid terminal worker status: {status}")
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worker_state
                SET status = ?, current_job_id = NULL, last_heartbeat_at = ?,
                    lease_expires_at = ?, stop_requested = 0, error = ?
                WHERE singleton = 1 AND worker_id = ?
                """,
                (status, now, now, error, worker_id),
            )
