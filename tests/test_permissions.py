"""Permission gating — in particular the plan-mode MCP hole.

MCP tool names appear in neither ``READ_TOOLS`` nor ``EDIT_TOOLS``. Before this
gate existed they fell through ``check_permission`` to the final ``allow``, so a
model in plan mode — a session the user believes is read-only — could call a
side-effecting MCP tool such as "send mail" or "update row".
"""

from __future__ import annotations

from collections.abc import Callable

from mini_claude.tools import check_permission


def _read_only(names: set[str]) -> Callable[[str], bool]:
    """Stand-in for ``McpManager.is_read_only_tool`` — no servers needed."""
    return lambda name: name in names


def test_plan_mode_blocks_a_side_effecting_mcp_tool() -> None:
    result = check_permission(
        "mcp__mail__send", {"to": "someone"}, "plan", mcp_read_only=_read_only(set())
    )

    assert result["action"] == "deny"
    assert "mcp__mail__send" in result["message"]


def test_plan_mode_allows_a_read_only_mcp_tool() -> None:
    result = check_permission(
        "mcp__amap__maps_weather",
        {"city": "大连"},
        "plan",
        mcp_read_only=_read_only({"mcp__amap__maps_weather"}),
    )

    assert result["action"] == "allow"


def test_plan_mode_fails_closed_without_a_read_only_probe() -> None:
    """A caller that forgets the probe must deny, never allow."""
    assert check_permission("mcp__mail__send", {}, "plan")["action"] == "deny"


def test_mcp_tools_are_untouched_outside_plan_mode() -> None:
    for mode in ("default", "acceptEdits", "bypassPermissions", "dontAsk"):
        assert check_permission("mcp__mail__send", {}, mode)["action"] == "allow"


def test_plan_mode_keeps_its_existing_verdicts_for_builtin_tools() -> None:
    assert check_permission("read_file", {"file_path": "x"}, "plan")["action"] == "allow"
    assert check_permission("write_file", {"file_path": "x"}, "plan")["action"] == "deny"
    assert check_permission("run_shell", {"command": "ls"}, "plan")["action"] == "deny"
