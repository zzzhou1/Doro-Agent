from pathlib import Path

from phm_mcp.admin import AdminService
from phm_mcp.admin_server import TOOLS, call_tool, handle


def test_admin_server_lists_training_control_tools(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    initialized = handle({"id": 1, "method": "initialize"}, service)
    assert initialized["result"]["serverInfo"]["name"] == "mini-claude-phm-admin"
    listed = handle({"id": 2, "method": "tools/list"}, service)
    assert len(listed["result"]["tools"]) == 7
    assert {tool["name"] for tool in TOOLS} == {
        "submit_training_job",
        "get_training_job",
        "list_training_jobs",
        "cancel_training_job",
        "promote_candidate_model",
        "rollback_phm_model",
        "get_training_control_status",
    }


def test_submit_rejects_missing_prepared_data_and_bad_params(tmp_path: Path) -> None:
    service = AdminService(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")
    missing = call_tool(service, "submit_training_job", {"model": "lstm"})
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
