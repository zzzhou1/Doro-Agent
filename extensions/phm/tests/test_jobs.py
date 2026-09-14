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
