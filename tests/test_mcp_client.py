from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from mini_claude.mcp_client import McpConnection, McpError, McpManager, McpTimeoutError


SERVER = Path(__file__).parent / "fixtures" / "mcp_server.py"

# Reads whatever arrives, then exits without ever answering.
STALL_SCRIPT = "import sys, time\nfor line in sys.stdin:\n    time.sleep(3600)\n"
# Reads one request, then exits — simulating a server that dies mid-call.
EXIT_SCRIPT = "import sys\nsys.stdin.readline()\nsys.exit(0)\n"


@pytest.mark.asyncio
async def test_python_mcp_server_round_trip() -> None:
    connection = McpConnection("test", sys.executable, [str(SERVER)])
    await connection.connect()
    process = connection._process
    try:
        await asyncio.wait_for(connection.initialize(), timeout=5)
        tools = await asyncio.wait_for(connection.list_tools(), timeout=5)
        assert [tool["name"] for tool in tools] == ["echo", "add"]
        assert await asyncio.wait_for(
            connection.call_tool("echo", {"text": "hello"}), timeout=5
        ) == "hello"
        assert await asyncio.wait_for(
            connection.call_tool("add", {"a": 2, "b": 3}), timeout=5
        ) == "5"
    finally:
        connection.close()
        if process is not None:
            await asyncio.wait_for(process.wait(), timeout=5)


@pytest.mark.asyncio
async def test_call_tool_times_out_on_a_stalled_server() -> None:
    """A server that never replies must not hang the agent forever."""
    connection = McpConnection(
        "slow", sys.executable, ["-c", STALL_SCRIPT], tool_timeout=0.5
    )
    await connection.connect()
    try:
        with pytest.raises(McpTimeoutError):
            await asyncio.wait_for(connection.call_tool("stall", {}), timeout=5)
        # The abandoned request slot must be gone, so a late reply can't
        # resolve a future nobody is waiting on.
        assert connection._pending == {}
    finally:
        await connection.aclose()


@pytest.mark.asyncio
async def test_call_tool_fails_fast_when_the_server_exits() -> None:
    """A server dying mid-call fails the request instead of waiting forever."""
    connection = McpConnection("dying", sys.executable, ["-c", EXIT_SCRIPT])
    await connection.connect()
    try:
        with pytest.raises(McpError, match="exited"):
            await asyncio.wait_for(connection.call_tool("whatever", {}), timeout=5)
    finally:
        await connection.aclose()


@pytest.mark.asyncio
async def test_manager_returns_an_error_string_instead_of_raising() -> None:
    """Tool failures degrade into result text so the agent turn survives."""
    connection = McpConnection(
        "slow", sys.executable, ["-c", STALL_SCRIPT], tool_timeout=0.5
    )
    await connection.connect()
    manager = McpManager()
    manager._connections["slow"] = connection
    try:
        result = await asyncio.wait_for(
            manager.call_tool("mcp__slow__stall", {}), timeout=5
        )
        assert result.startswith("[mcp] tool 'stall' did not respond within 0.5s")
        assert await manager.call_tool("mcp__ghost__x", {}) == "[mcp] Server 'ghost' is not connected."
        malformed = await manager.call_tool("not_mcp", {})
        assert malformed.startswith("[mcp] Invalid MCP tool name")
    finally:
        await connection.aclose()


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_cross_talk() -> None:
    """Parallel calls share one stdio pipe — every reply must match its request."""
    connection = McpConnection("test", sys.executable, [str(SERVER)])
    await connection.connect()
    try:
        await asyncio.wait_for(connection.initialize(), timeout=5)
        payloads = [f"token-{i}" for i in range(25)]
        results = await asyncio.wait_for(
            asyncio.gather(*(connection.call_tool("echo", {"text": p}) for p in payloads)),
            timeout=10,
        )
        assert results == payloads
    finally:
        await connection.aclose()


@pytest.mark.asyncio
async def test_server_exit_releases_a_concurrent_call() -> None:
    """A dying server must fail in-flight calls, not strand them."""
    connection = McpConnection("dying", sys.executable, ["-c", EXIT_SCRIPT])
    await connection.connect()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                connection.call_tool("a", {}),
                connection.call_tool("b", {}),
                return_exceptions=True,
            ),
            timeout=5,
        )
        assert all(isinstance(r, McpError) for r in results), results
    finally:
        await connection.aclose()


def test_concurrency_safe_is_opt_in_per_server() -> None:
    manager = McpManager()
    manager._connections["ro"] = McpConnection("ro", sys.executable, [], read_only=True)
    manager._connections["rw"] = McpConnection("rw", sys.executable, [], read_only=False)

    assert manager.is_concurrency_safe("mcp__ro__lookup") is True
    assert manager.is_concurrency_safe("mcp__rw__lookup") is False
    assert manager.is_concurrency_safe("mcp__ghost__lookup") is False
    assert manager.is_concurrency_safe("read_file") is False
    assert manager.is_concurrency_safe("mcp__malformed") is False


def test_get_tool_definitions_skips_names_the_api_would_reject() -> None:
    manager = McpManager()
    manager._tools = [
        {"name": "ok", "description": "d", "inputSchema": None, "serverName": "amap"},
        {"name": "has.dot", "description": "d", "inputSchema": None, "serverName": "amap"},
        {"name": "x" * 80, "description": "d", "inputSchema": None, "serverName": "amap"},
    ]
    assert [d["name"] for d in manager.get_tool_definitions()] == ["mcp__amap__ok"]


def test_max_qps_config_becomes_a_min_interval() -> None:
    """A documented requests-per-second cap must translate into a start gap."""
    manager = McpManager()

    by_qps = manager._build_connection("svc", {"command": "python", "maxQps": 3})
    assert by_qps.min_interval == pytest.approx(1 / 3)

    # An explicit interval wins, so a user can hand-tune a bursty server.
    explicit = manager._build_connection(
        "svc2", {"command": "python", "minInterval": 0.5, "maxQps": 3}
    )
    assert explicit.min_interval == 0.5

    unthrottled = manager._build_connection("svc3", {"command": "python"})
    assert unthrottled.min_interval == 0.0


@pytest.mark.asyncio
async def test_min_interval_spaces_out_concurrent_call_starts() -> None:
    """Three calls at maxQps=3 must not fire inside the same instant."""
    connection = McpConnection("spaced", sys.executable, [], min_interval=0.2)
    started = time.monotonic()
    await asyncio.gather(*(connection._throttle() for _ in range(3)))

    # Two gaps of 0.2s — generous lower bound so a loaded CI box can't flake.
    elapsed = time.monotonic() - started
    assert elapsed >= 0.35, f"calls started too close together: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_no_throttle_by_default() -> None:
    """Servers without a QPS cap must not pay for a limiter they don't need."""
    connection = McpConnection("plain", sys.executable, [])
    started = time.monotonic()
    await asyncio.gather(*(connection._throttle() for _ in range(20)))
    assert time.monotonic() - started < 0.1


def test_build_connection_selects_the_transport() -> None:
    manager = McpManager()

    stdio = manager._build_connection("local", {"command": "python", "args": ["s.py"]})
    assert isinstance(stdio, McpConnection)
    assert stdio.transport == "stdio"
    assert stdio.tool_timeout == 60.0

    remote = manager._build_connection(
        "remote", {"url": "https://host.test/mcp", "readOnly": True, "toolTimeout": 5}
    )
    assert remote.read_only is True
    assert remote.tool_timeout == 5.0
    assert remote.transport == "http"

    with pytest.raises(ValueError):
        manager._build_connection("bad__name", {"command": "python"})
    with pytest.raises(ValueError):
        manager._build_connection("badurl", {"url": "ftp://host/mcp"})


def test_settings_json_without_mcp_servers_is_not_a_server_map(tmp_path, capsys) -> None:
    """A real settings.json holds env/modelSettings — not MCP servers."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"env": {"A": "1"}, "modelSettings": {"temperature": 0.2}}),
        encoding="utf-8",
    )
    target: dict[str, dict] = {}
    McpManager()._merge_config_file(path, target)

    assert target == {}
    assert capsys.readouterr().out == ""


def test_broken_entries_are_reported_only_inside_mcp_servers(tmp_path, capsys) -> None:
    path = tmp_path / ".mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {
            "broken": {"description": "no command and no url"},
            "ok": {"command": "python"},
        }}),
        encoding="utf-8",
    )
    target: dict[str, dict] = {}
    McpManager()._merge_config_file(path, target)

    assert list(target) == ["ok"]
    assert "Ignoring 'broken'" in capsys.readouterr().out


def test_serverurl_is_accepted_as_an_alias_for_url(tmp_path) -> None:
    path = tmp_path / ".mcp.json"
    path.write_text(
        json.dumps({"remote": {"serverUrl": "https://host.test/mcp"}}),
        encoding="utf-8",
    )
    target: dict[str, dict] = {}
    McpManager()._merge_config_file(path, target)

    assert target["remote"]["url"] == "https://host.test/mcp"
