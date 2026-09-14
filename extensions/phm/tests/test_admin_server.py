from pathlib import Path

import pytest
from phm_mcp.admin import AdminService
from phm_mcp.admin_server import TOOLS, call_tool, handle


def test_admin_server_lists_training_control_tools(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    initialized = handle({"id": 1, "method": "initialize"}, service)
    assert initialized["result"]["serverInfo"]["name"] == "doro-phm-admin"
    listed = handle({"id": 2, "method": "tools/list"}, service)
    assert len(listed["result"]["tools"]) == 10
    assert {tool["name"] for tool in TOOLS} == {
        "submit_training_job",
        "ensure_training_worker",
        "get_training_worker_status",
        "stop_training_worker",
        "get_training_job",
        "list_training_jobs",
        "cancel_training_job",
        "promote_candidate_model",
        "rollback_phm_model",
        "get_training_control_status",
    }


def test_submit_rejects_missing_prepared_data_and_bad_params(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    missing = call_tool(
        service,
        "submit_training_job",
        {"model": "lstm", "preset_confirmed": True, "auto_start_worker": False},
    )
    assert missing["isError"] is True
    assert "phm prepare" in missing["content"][0]["text"]

    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "fd001.npz").touch()
    invalid = call_tool(
        service,
        "submit_training_job",
        {"model": "lstm", "epochs": 2, "patience": 3},
    )
    assert invalid["isError"] is True
    assert "patience" in invalid["content"][0]["text"]


def test_presets_resolve_params_and_auto_start_worker(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    processed = tmp_path / "data" / "processed" / "fd001.npz"
    processed.parent.mkdir(parents=True)
    processed.touch()

    class FakeSupervisor:
        started = 0

        def ensure_worker(self):
            self.started += 1
            return {"action": "started", "alive": True, "pid": 123}

        def status(self):
            return {"status": "stopped", "alive": False, "queue_depth": 0}

    supervisor = FakeSupervisor()
    service.supervisor = supervisor

    standard = service.submit_training_job(model="lstm", version="preset-v1", preset_confirmed=True)
    assert standard["preset"] == "standard"
    assert standard["parameter_source"] == "confirmed_preset"
    assert standard["parameter_overrides"] == {}
    assert standard["params"] == {
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "patience": 5,
        "seed": 42,
        "device": "auto",
        "amp": False,
    }
    assert standard["worker"]["action"] == "started"
    assert supervisor.started == 1

    overridden = service.submit_training_job(
        model="transformer",
        version="preset-v2",
        epochs=3,
        auto_start_worker=False,
    )
    assert overridden["parameter_source"] == "preset_with_overrides"
    assert overridden["params"]["epochs"] == 3
    assert overridden["params"]["patience"] == 3
    assert overridden["parameter_overrides"] == {"epochs": 3}
    assert overridden["worker"]["action"] == "not_requested"


def test_invalid_preset_is_rejected(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    with pytest.raises(ValueError, match="preset must be"):
        service.submit_training_job(model="lstm", preset="unknown")


def test_unconfirmed_default_parameters_are_rejected(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    with pytest.raises(ValueError, match="Ask the user to confirm"):
        service.submit_training_job(model="lstm")
