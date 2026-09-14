"""REPL command surface for MCP status and diagnostics.

``/mcp``, ``/config`` and ``/doctor`` exist so a failure is diagnosable from
inside the REPL instead of from the startup log. These tests drive the real
dispatch path (``run_repl``) so a handler that is registered but wired to the
wrong command still fails.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from doro import __main__ as main
from doro.agent import Agent
from doro.mcp_client import McpServerStatus
from doro.status import format_status_toolbar


def _agent() -> Agent:
    with patch("doro.agent.openai.AsyncOpenAI"):
        return Agent(
            backend="openai",
            api_key="sk-test-key-1234567890",
            model_source="--model",
            effort_source="--effort",
        )


def _isolate_mcp_config(monkeypatch, tmp_path) -> None:
    empty = [tmp_path / ".mcp.json"]
    monkeypatch.setattr(
        "doro.mcp_client.config_search_paths", lambda cwd=None: list(empty)
    )
    monkeypatch.setattr(
        "doro.diagnostics.config_search_paths", lambda cwd=None: list(empty)
    )


async def _run(monkeypatch, agent: Agent, lines: list[str], panels: list, errors: list):
    def record_panel(title, content):
        panels.append((title, list(content)))

    monkeypatch.setattr(main, "print_panel", record_panel)
    monkeypatch.setattr(main, "print_error", errors.append)
    monkeypatch.setattr(main, "print_info", lambda _msg: None)
    monkeypatch.setattr(agent, "start_mcp_background", lambda: None)

    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=lines)
    with (
        patch("doro.__main__.signal.signal"),
        patch("doro.__main__.print_welcome"),
    ):
        await main.run_repl(agent, prompt_session=prompt_session)


@pytest.mark.asyncio
async def test_mcp_command_lists_nothing_configured(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    panels: list = []

    await _run(monkeypatch, agent, ["/mcp", "exit"], panels, [])

    assert panels[0][0] == "MCP servers"
    assert any("No MCP servers configured" in line for line in panels[0][1])


@pytest.mark.asyncio
async def test_mcp_command_reports_mixed_server_states(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    manager = agent.mcp_manager
    manager._configs = {"amap": {}, "weather": {}}
    manager._states = {
        "amap": McpServerStatus(
            name="amap", status="connected", transport="stdio", tool_count=12
        ),
        "weather": McpServerStatus(
            name="weather", status="failed", error="connection refused", attempts=2
        ),
    }
    manager._states["weather"].next_retry_at = 0.0
    panels: list = []

    await _run(monkeypatch, agent, ["/mcp", "exit"], panels, [])

    rendered = "\n".join(panels[0][1])
    assert "1/2 connected" in rendered
    assert "connected" in rendered and "12 tools" in rendered
    assert "connection refused" in rendered


@pytest.mark.asyncio
async def test_mcp_tools_lists_each_server(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    agent.mcp_manager._configs = {"amap": {}}
    agent.mcp_manager._states = {
        "amap": McpServerStatus(name="amap", status="connected", tool_count=2)
    }
    agent.mcp_manager._tools = [
        {"serverName": "amap", "name": "maps_weather"},
        {"serverName": "amap", "name": "maps_geo"},
    ]
    panels: list = []

    await _run(monkeypatch, agent, ["/mcp tools", "exit"], panels, [])

    assert panels[0][0] == "MCP tools"
    rendered = "\n".join(panels[0][1])
    assert "maps_weather" in rendered and "maps_geo" in rendered
    assert "2 tools across 1 server" in rendered


@pytest.mark.asyncio
async def test_mcp_tools_is_empty_before_any_connection(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    panels: list = []

    await _run(monkeypatch, agent, ["/mcp tools", "exit"], panels, [])

    assert any("No MCP tools discovered" in line for line in panels[0][1])


@pytest.mark.asyncio
async def test_mcp_reconnect_calls_the_manager(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    agent.reconnect_mcp = AsyncMock(return_value=True)
    panels: list = []

    await _run(monkeypatch, agent, ["/mcp reconnect", "exit"], panels, [])

    agent.reconnect_mcp.assert_awaited_once_with()
    assert panels[0][0] == "MCP servers"


@pytest.mark.asyncio
async def test_unknown_mcp_subcommand_reports_an_error(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    errors: list = []

    await _run(monkeypatch, agent, ["/mcp bogus", "exit"], [], errors)

    assert errors and "Unknown /mcp subcommand" in errors[0]


@pytest.mark.asyncio
async def test_config_and_doctor_dispatch_to_their_reports(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    panels: list = []

    await _run(monkeypatch, agent, ["/config", "/doctor", "exit"], panels, [])

    assert [title for title, _ in panels] == ["Configuration", "Doctor"]


@pytest.mark.asyncio
async def test_config_report_hides_the_api_key(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()
    panels: list = []

    await _run(monkeypatch, agent, ["/config", "exit"], panels, [])

    rendered = "\n".join(panels[0][1])
    assert "sk-test-key-1234567890" not in rendered
    assert "sk-t" in rendered


# ─── status bar ─────────────────────────────────────────────


def _snapshot(**overrides) -> dict:
    value = {
        "cwd": r"F:\WorkSpace",
        "context_used": 0,
        "context_window": 128000,
        "backend": "openai",
        "model": "gpt-5.6-sol",
    }
    value.update(overrides)
    return value


def test_toolbar_shows_mcp_counts_when_present() -> None:
    line = format_status_toolbar(_snapshot(mcp_connected=2, mcp_total=3), width=120)

    assert "mcp: 2/3" in line


def test_toolbar_omits_mcp_segment_without_counts() -> None:
    assert "mcp" not in format_status_toolbar(_snapshot(), width=120)


def test_toolbar_omits_mcp_segment_when_nothing_is_configured() -> None:
    """A configured-nothing session must not spend status-bar width on ``mcp: 0/0``."""
    line = format_status_toolbar(_snapshot(mcp_connected=0, mcp_total=0), width=120)
    assert "mcp" not in line


def test_snapshot_carries_mcp_counts() -> None:
    agent = _agent()
    agent.mcp_manager._configs = {"amap": {}}
    agent.mcp_manager._states = {
        "amap": McpServerStatus(name="amap", status="connected", tool_count=3)
    }

    snapshot = agent.get_status_snapshot()

    assert (snapshot["mcp_connected"], snapshot["mcp_total"]) == (1, 1)
