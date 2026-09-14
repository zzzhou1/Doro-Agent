from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

import mini_claude.mcp_client as mcp_client
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
        assert (
            await asyncio.wait_for(connection.call_tool("echo", {"text": "hello"}), timeout=5)
            == "hello"
        )
        assert (
            await asyncio.wait_for(connection.call_tool("add", {"a": 2, "b": 3}), timeout=5) == "5"
        )
    finally:
        connection.close()
        if process is not None:
            await asyncio.wait_for(process.wait(), timeout=5)


@pytest.mark.asyncio
async def test_call_tool_times_out_on_a_stalled_server() -> None:
    """A server that never replies must not hang the agent forever."""
    connection = McpConnection("slow", sys.executable, ["-c", STALL_SCRIPT], tool_timeout=0.5)
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
    connection = McpConnection("slow", sys.executable, ["-c", STALL_SCRIPT], tool_timeout=0.5)
    await connection.connect()
    manager = McpManager()
    manager._connections["slow"] = connection
    try:
        result = await asyncio.wait_for(manager.call_tool("mcp__slow__stall", {}), timeout=5)
        assert result.startswith("[mcp] tool 'stall' did not respond within 0.5s")
        assert (
            await manager.call_tool("mcp__ghost__x", {}) == "[mcp] Server 'ghost' is not connected."
        )
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


def test_read_only_flag_is_shared_by_the_plan_mode_gate() -> None:
    """Plan mode and concurrency read the same ``readOnly`` statement."""
    manager = McpManager()
    manager._connections["ro"] = McpConnection("ro", sys.executable, [], read_only=True)
    manager._connections["rw"] = McpConnection("rw", sys.executable, [], read_only=False)

    assert manager.is_read_only_tool("mcp__ro__send") is True
    assert manager.is_read_only_tool("mcp__rw__send") is False
    assert manager.is_read_only_tool("mcp__ghost__send") is False
    assert manager.is_read_only_tool("read_file") is False
    assert manager.is_read_only_tool("mcp__malformed") is False


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
    assert stdio.cwd is None

    with_cwd = manager._build_connection("local-cwd", {"command": "python", "cwd": "/tmp/project"})
    assert isinstance(with_cwd, McpConnection)
    assert with_cwd.cwd == "/tmp/project"

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
        json.dumps(
            {
                "mcpServers": {
                    "broken": {"description": "no command and no url"},
                    "ok": {"command": "python"},
                }
            }
        ),
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


def test_relative_stdio_cwd_is_resolved_from_mcp_config_project(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = project / ".mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python", "cwd": "."}}}),
        encoding="utf-8",
    )
    target: dict[str, dict] = {}

    McpManager()._merge_config_file(path, target)

    assert target["demo"]["cwd"] == str(project.resolve())


def test_settings_relative_stdio_cwd_is_resolved_from_project(tmp_path) -> None:
    project = tmp_path / "project"
    config_dir = project / ".claude"
    config_dir.mkdir(parents=True)
    path = config_dir / "settings.json"
    path.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python", "cwd": "."}}}),
        encoding="utf-8",
    )
    target: dict[str, dict] = {}

    McpManager()._merge_config_file(path, target)

    assert target["demo"]["cwd"] == str(project.resolve())


def test_invalid_stdio_cwd_is_rejected(tmp_path, capsys) -> None:
    path = tmp_path / ".mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python", "cwd": ""}}}),
        encoding="utf-8",
    )
    target: dict[str, dict] = {}

    McpManager()._merge_config_file(path, target)

    assert target == {}
    assert "'cwd' must be a non-empty string" in capsys.readouterr().out


def test_install_root_config_is_loaded_outside_working_directory(tmp_path, monkeypatch) -> None:
    install_root = tmp_path / "install"
    home = tmp_path / "home"
    working = tmp_path / "working"
    install_root.mkdir()
    home.mkdir()
    working.mkdir()
    (install_root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_client, "INSTALL_ROOT", install_root)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: working))

    configs = McpManager()._load_configs()

    assert list(configs) == ["demo"]


@pytest.mark.asyncio
async def test_failed_server_connection_is_retried(monkeypatch) -> None:
    manager = McpManager()
    attempts = 0

    class FailingConnection:
        connect_timeout = 0.1
        read_only = False
        transport = "stdio"

        async def connect(self):
            nonlocal attempts
            attempts += 1
            raise McpError("temporary failure")

        async def aclose(self):
            pass

    monkeypatch.setattr(
        manager,
        "_load_configs",
        lambda: {"demo": {"command": "python"}},
    )
    monkeypatch.setattr(
        manager,
        "_build_connection",
        lambda _name, _cfg: FailingConnection(),
    )

    assert await manager.load_and_connect() is False
    assert manager._connected is False
    # Backoff blocks an immediate automatic retry; a forced reconnect (manual
    # /mcp reconnect, or the background loop after the deadline) goes through.
    assert await manager.load_and_connect(force=True) is False
    assert manager._connected is False
    assert attempts == 2


class _FakeConnection:
    """Configurable stand-in for _McpConnection — no subprocess needed."""

    def __init__(self, delay: float = 0.0, fail: bool = False, tool_count: int = 0):
        self.delay = delay
        self.fail = fail
        self.tool_count = tool_count
        self.connect_timeout = 5
        self.read_only = False
        self.transport = "stdio"
        self.connect_calls = 0

    async def connect(self):
        self.connect_calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise McpError("temporary failure")

    async def initialize(self):
        return {}

    async def list_tools(self):
        return [{"name": f"t{i}", "serverName": "fake"} for i in range(self.tool_count)]

    async def aclose(self):
        pass


def _patch_servers(monkeypatch, manager: McpManager, fakes: dict[str, _FakeConnection]):
    monkeypatch.setattr(
        manager, "_load_configs", lambda: {name: {"command": "x"} for name in fakes}
    )
    monkeypatch.setattr(manager, "_build_connection", lambda name, _cfg: fakes[name])


@pytest.mark.asyncio
async def test_servers_connect_in_parallel(monkeypatch) -> None:
    """Two 0.3s handshakes must overlap, not queue up one after the other."""
    manager = McpManager()
    _patch_servers(
        monkeypatch,
        manager,
        {
            "a": _FakeConnection(delay=0.3),
            "b": _FakeConnection(delay=0.3),
        },
    )

    started = time.monotonic()
    assert await manager.load_and_connect() is True
    elapsed = time.monotonic() - started
    assert elapsed < 0.55, f"connections look serial: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_failed_server_backs_off_until_the_deadline(monkeypatch) -> None:
    manager = McpManager()
    fake = _FakeConnection(fail=True)
    _patch_servers(monkeypatch, manager, {"demo": fake})

    assert await manager.load_and_connect() is False
    assert fake.connect_calls == 1

    # Inside the backoff window a normal pass skips the server entirely.
    assert await manager.load_and_connect() is False
    assert fake.connect_calls == 1

    # Once the deadline passes, the next pass retries it.
    manager._states["demo"].next_retry_at = time.monotonic() - 1
    assert await manager.load_and_connect() is False
    assert fake.connect_calls == 2


@pytest.mark.asyncio
async def test_invalid_config_is_skipped_permanently(monkeypatch) -> None:
    """A bad config is terminal (not retried), and doesn't block 'connected'."""
    manager = McpManager()
    monkeypatch.setattr(manager, "_load_configs", lambda: {"bad": {"command": "x"}})

    def bad_builder(_name, _cfg):
        raise ValueError("name may only contain letters")

    monkeypatch.setattr(manager, "_build_connection", bad_builder)

    assert await manager.load_and_connect() is True
    state = manager._states["bad"]
    assert state.status == "skipped"
    assert state.next_retry_at == float("inf")
    # The background retry loop must not fire for a skipped server.
    manager._schedule_retry_loop()
    assert manager._retry_task is None


@pytest.mark.asyncio
async def test_wait_ready_uses_the_background_task_and_stays_cheap(monkeypatch) -> None:
    manager = McpManager()
    fake = _FakeConnection(tool_count=2)
    _patch_servers(monkeypatch, manager, {"demo": fake})

    manager.start_background()
    assert manager._connect_task is not None

    assert await manager.wait_ready() is True
    assert manager._states["demo"].status == "connected"
    assert manager._states["demo"].tool_count == 2

    # Later calls (every chat turn) must not reconnect or wait on anything.
    started = time.monotonic()
    assert await manager.wait_ready() is True
    assert time.monotonic() - started < 0.1
    assert fake.connect_calls == 1


@pytest.mark.asyncio
async def test_background_retry_loop_reconnects_after_backoff(monkeypatch) -> None:
    monkeypatch.setattr(mcp_client, "_RETRY_DELAYS", (0.05,))
    manager = McpManager()
    fake = _FakeConnection(fail=True)
    _patch_servers(monkeypatch, manager, {"demo": fake})

    assert await manager.load_and_connect() is False
    fake.fail = False  # server "recovers" before the retry fires

    manager._schedule_retry_loop()
    assert manager._retry_task is not None
    await asyncio.wait_for(manager._retry_task, timeout=5)

    assert manager._connected is True
    assert manager._states["demo"].status == "connected"
    assert fake.connect_calls == 2


@pytest.mark.asyncio
async def test_server_statuses_cover_mixed_outcomes(monkeypatch) -> None:
    manager = McpManager()
    _patch_servers(
        monkeypatch,
        manager,
        {
            "up": _FakeConnection(tool_count=3),
            "down": _FakeConnection(fail=True),
        },
    )
    assert await manager.load_and_connect() is False

    statuses = {s["name"]: s for s in manager.server_statuses()}
    assert statuses["up"]["status"] == "connected"
    assert statuses["up"]["tool_count"] == 3
    assert statuses["up"]["transport"] == "stdio"
    assert statuses["down"]["status"] == "failed"
    assert statuses["down"]["error"] == "temporary failure"
    assert statuses["down"]["retry_in"] > 0
    assert manager.status_counts() == (1, 2)
