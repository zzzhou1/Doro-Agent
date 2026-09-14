from pathlib import Path

import pytest
from phm_mcp.jobs import JobStore


def test_job_queue_claim_progress_and_cancel(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    queued = store.enqueue("lstm", "v2", {"epochs": 2})
    assert queued["status"] == "queued"

    running = store.claim_next(1234)
    assert running is not None
    assert running["id"] == queued["id"]
    assert running["status"] == "running"
    assert running["worker_pid"] == 1234

    store.update_progress(queued["id"], {"epoch": 1, "validation_mse": 10.0})
    assert store.require(queued["id"])["progress"]["epoch"] == 1
    cancelling = store.request_cancel(queued["id"])
    assert cancelling["status"] == "cancelling"
    assert store.cancellation_requested(queued["id"]) is True
    with pytest.raises(ValueError, match="Cannot complete"):
        store.mark_succeeded(queued["id"], tmp_path / "candidate", {})
    assert store.require(queued["id"])["status"] == "cancelling"
    assert store.mark_cancelled(queued["id"])["status"] == "cancelled"


def test_queued_job_can_be_cancelled_without_a_worker(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.enqueue("transformer", "experiment-1", {"epochs": 2})
    cancelled = store.request_cancel(job["id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["finished_at"] is not None
    assert store.claim_next(1) is None
    with pytest.raises(ValueError, match="Cannot cancel"):
        store.request_cancel(job["id"])


def test_fail_interrupted_marks_only_running_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    running = store.enqueue("lstm", "v2", {"epochs": 2})
    queued = store.enqueue("lstm", "v3", {"epochs": 2})
    store.claim_next(123)
    assert store.fail_interrupted() == 1
    assert store.require(running["id"])["status"] == "failed"
    assert store.require(queued["id"])["status"] == "queued"


def test_queue_position_and_training_metadata(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    first = store.enqueue(
        "lstm",
        "v2",
        {"epochs": 20},
        preset="standard",
        parameter_source="preset",
        auto_start_worker=True,
        parameter_overrides={"epochs": 20},
    )
    second = store.enqueue("transformer", "v2", {"epochs": 20})

    assert first["preset"] == "standard"
    assert first["parameter_source"] == "preset"
    assert first["auto_start_worker"] is True
    assert first["parameter_overrides"] == {"epochs": 20}
    assert store.queue_position(first["id"]) == 1
    assert store.queue_position(second["id"]) == 2
    assert store.queue_depth() == 2
    store.claim_next(10)
    assert store.queue_position(first["id"]) is None
    assert store.queue_position(second["id"]) == 1


def test_single_worker_lease_heartbeat_and_stop(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    first = store.reserve_worker_start("worker-a")
    second = store.reserve_worker_start("worker-b")

    assert first["reserved"] is True
    assert second["reserved"] is False
    assert second["worker"]["worker_id"] == "worker-a"
    store.record_worker_process("worker-a", 1234)
    assert store.activate_worker("worker-a", 1234) is True
    assert store.heartbeat_worker("worker-a", 1234, "busy", "job-1") is True
    status = store.worker_status()
    assert status["alive"] is True
    assert status["status"] == "busy"
    assert status["current_job_id"] == "job-1"
    assert status["pid"] == 1234

    stopping = store.request_worker_stop()
    assert stopping["action"] == "stop_requested"
    assert stopping["stop_requested"] is True
    assert store.worker_stop_requested("worker-a") is True
    store.release_worker("worker-a", "stopped")
    assert store.worker_status()["alive"] is False


def test_stop_requested_during_start_is_preserved(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    assert store.reserve_worker_start("worker-a")["reserved"] is True
    assert store.request_worker_stop()["action"] == "stop_requested"
    assert store.activate_worker("worker-a", 1234) is True
    assert store.worker_stop_requested("worker-a") is True


def test_job_store_migrates_existing_database(tmp_path: Path) -> None:
    import sqlite3

    database = tmp_path / "jobs.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE training_jobs (
                id TEXT PRIMARY KEY, model TEXT NOT NULL,
                candidate_version TEXT NOT NULL, status TEXT NOT NULL,
                params_json TEXT NOT NULL, progress_json TEXT, metrics_json TEXT,
                candidate_path TEXT, error TEXT, worker_pid INTEGER,
                created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

    store = JobStore(database)
    migrated = store.enqueue("lstm", "v2", {"epochs": 1})
    assert migrated["preset"] == "legacy"
    assert migrated["parameter_source"] == "legacy"
    assert migrated["auto_start_worker"] is False
    assert migrated["parameter_overrides"] == {}
