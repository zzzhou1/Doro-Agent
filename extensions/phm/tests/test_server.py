from pathlib import Path

from phm_mcp.inference import InferenceService
from phm_mcp.server import TOOLS, call_tool, handle


def test_server_lists_six_read_only_tools() -> None:
    assert len(TOOLS) == 6
    assert {tool["name"] for tool in TOOLS} == {
        "list_phm_models",
        "inspect_engine",
        "predict_rul",
        "compare_rul_models",
        "get_model_metrics",
        "get_degradation_evidence",
    }


def test_initialize_and_tool_list_protocol(tmp_path: Path) -> None:
    service = InferenceService(tmp_path / "data", tmp_path / "artifacts")
    initialized = handle({"id": 1, "method": "initialize"}, service)
    assert initialized["result"]["serverInfo"]["name"] == "mini-claude-phm"
    listed = handle({"id": 2, "method": "tools/list"}, service)
    assert len(listed["result"]["tools"]) == 6


def test_missing_prepared_data_is_a_tool_error(tmp_path: Path) -> None:
    service = InferenceService(tmp_path / "data", tmp_path / "artifacts")
    result = call_tool(service, "inspect_engine", {"unit_id": 1})
    assert result["isError"] is True
    assert "phm prepare" in result["content"][0]["text"]


def test_unknown_tool_is_a_tool_error(tmp_path: Path) -> None:
    service = InferenceService(tmp_path / "data", tmp_path / "artifacts")
    result = call_tool(service, "unknown", {})
    assert result["isError"] is True
