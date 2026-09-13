"""
MCP Client — connects to MCP servers over stdio or HTTP/SSE, discovers and
forwards tool calls. Raw JSON-RPC 2.0, no MCP SDK dependency.

Config sources (read in order — later files win on name collisions):
  1. ~/.claude/settings.json
  2. ./.claude/settings.json
  3. ./.mcp.json

stdio server — the server runs as a local subprocess:
  "mcpServers": {
    "amap": { "command": "npx.cmd", "args": ["-y", "@amap/amap-maps-mcp-server"],
              "env": { "AMAP_MAPS_API_KEY": "..." } }
  }

Remote server — Streamable HTTP (2025-03-26), with automatic fallback to the
older HTTP+SSE transport (2024-11-05):
  "mcpServers": {
    "remote": { "url": "https://host/mcp",
                "headers": { "Authorization": "Bearer ..." },
                "transport": "auto" }        # "auto" | "http" | "sse"
  }

Per-server tuning (optional, both transports):
  "connectTimeout": 15     seconds for connect + initialize + tools/list
  "toolTimeout":    60     seconds for a single tools/call
  "readOnly":       true   mark the server's tools as parallel-safe

Each MCP tool is exposed to the model as "mcp__serverName__toolName".
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urljoin


IS_WIN = sys.platform == "win32"

DEFAULT_CONNECT_TIMEOUT = 15.0
DEFAULT_TOOL_TIMEOUT = 60.0

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "mini-claude", "version": "1.0.0"}

# Tool names are forwarded verbatim to the model API, which only accepts a
# conservative character set and caps the length.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_PREFIXED_TOOL_LEN = 64

# Status codes that mean "this endpoint is not a Streamable HTTP server",
# which is our cue to retry over legacy HTTP+SSE.
_SSE_FALLBACK_STATUS = frozenset({404, 405, 406, 415})


# ─── Errors ──────────────────────────────────────────────────


class McpError(RuntimeError):
    """An MCP server reported an error, died, or misbehaved."""


class McpTimeoutError(McpError):
    """An MCP request exceeded its timeout budget."""


class _HttpStatusError(McpError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ─── JSON-RPC / SSE helpers ──────────────────────────────────


def _loads(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    return json.loads(raw)


def _unwrap(msg: dict) -> Any:
    """Turn a JSON-RPC response object into a result, or raise on error."""
    if "error" in msg:
        err = msg["error"] or {}
        raise McpError(f"MCP error {err.get('code')}: {err.get('message')}")
    return msg.get("result")


def _find_response(payload: Any, req_id: int) -> Any:
    """Locate the response matching req_id in a single or batched payload."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("id") == req_id:
                return _unwrap(item)
        return None
    if isinstance(payload, dict):
        return _unwrap(payload)
    return None


def _extract_text(result: Any) -> str:
    """Flatten an MCP tools/call result into plain text."""
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts = [
            c.get("text", "")
            for c in result["content"]
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p)
        # MCP signals tool failures with isError rather than a JSON-RPC error.
        if result.get("isError"):
            return f"[mcp error] {text or 'tool reported an error'}"
        return text
    return json.dumps(result, ensure_ascii=False)


async def _iter_sse(response) -> AsyncIterator[tuple[str, str]]:
    """Yield (event, data) pairs from an SSE response body."""
    event: str | None = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield event or "message", "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event or "message", "\n".join(data)


def _require_httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx ships with the SDKs
        raise McpError(
            "HTTP/SSE MCP servers need the 'httpx' package (pip install httpx)"
        ) from exc
    return httpx


# ─── Connection base ─────────────────────────────────────────


class _McpConnection:
    """MCP handshake and tool plumbing shared by every transport."""

    transport = "?"

    def __init__(
        self,
        server_name: str,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        read_only: bool = False,
        min_interval: float = 0.0,
    ):
        self.server_name = server_name
        self.connect_timeout = connect_timeout
        self.tool_timeout = tool_timeout
        self.read_only = read_only
        self.min_interval = min_interval
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._rate_lock = asyncio.Lock()
        self._last_call_start = 0.0

    # ── transport hooks (overridden per transport) ──

    async def connect(self) -> None:
        raise NotImplementedError

    async def _request(self, method: str, params: dict | None = None) -> Any:
        raise NotImplementedError

    async def _notify(self, method: str, params: dict | None = None) -> None:
        raise NotImplementedError

    # ── shared protocol ──

    def _resolve(self, msg: dict) -> None:
        """Complete the pending future for an incoming JSON-RPC response."""
        msg_id = msg.get("id")
        if msg_id is None:
            return
        fut = self._pending.pop(msg_id, None)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg["error"] or {}
            fut.set_exception(McpError(f"MCP error {err.get('code')}: {err.get('message')}"))
        else:
            fut.set_result(msg.get("result"))

    def _fail_pending(self, error: BaseException) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(error)
        self._pending.clear()

    async def initialize(self) -> dict:
        """Perform the MCP initialize handshake."""
        result = await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": dict(CLIENT_INFO),
        })
        await self._notify("notifications/initialized")
        return result if isinstance(result, dict) else {}

    async def list_tools(self) -> list[dict]:
        """Discover the tools this server exposes."""
        result = await self._request("tools/list")
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            return []
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema"),
                "serverName": self.server_name,
            }
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]

    async def _throttle(self) -> None:
        """Space out tool-call starts so a burst can't exceed the server's QPS cap.

        Many hosted servers (AMap, for one) rate-limit per API key — typically a
        handful of requests per second across a sliding window. Parallel calls
        are fine on our side but the upstream rejects them. A minimum gap between
        consecutive request *starts* keeps any 1-second window under the cap while
        still letting the requests overlap once they are in flight.
        """
        if self.min_interval <= 0:
            return
        async with self._rate_lock:
            wait = self._last_call_start + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_start = time.monotonic()

    async def call_tool(self, name: str, args: dict, timeout: float | None = None) -> str:
        """Call a tool, bounded by a timeout so a hung server can't stall the agent."""
        budget = self.tool_timeout if timeout is None else timeout
        # Throttle before starting the clock: waiting on our own rate limiter
        # must not eat into the budget reserved for the server's response.
        await self._throttle()
        try:
            result = await asyncio.wait_for(
                self._request("tools/call", {"name": name, "arguments": args or {}}),
                timeout=budget,
            )
        except asyncio.TimeoutError as exc:
            raise McpTimeoutError(f"tool '{name}' did not respond within {budget:g}s") from exc
        return _extract_text(result)

    # ── teardown ──

    def close(self) -> None:
        """Synchronous best-effort cleanup. Rejects any in-flight requests."""
        self._fail_pending(McpError(f"MCP server '{self.server_name}' closed"))

    async def aclose(self) -> None:
        """Graceful cleanup: wait for the transport to actually finish."""
        self.close()


# ─── stdio transport ─────────────────────────────────────────


class McpConnection(_McpConnection):
    """Local server: spawn a subprocess, talk newline-delimited JSON-RPC over stdio."""

    transport = "stdio"

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(server_name, **kwargs)
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        merged_env = {**os.environ, **self.env}
        kwargs: dict[str, Any] = {}
        if not IS_WIN:
            # Own process group, so _terminate() can kill the whole tree without
            # taking this process down with it.
            kwargs["start_new_session"] = True
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            **kwargs,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    @staticmethod
    def _terminate(proc: asyncio.subprocess.Process) -> None:
        """Kill the server *and its children*.

        A `npx.cmd` server actually runs as `cmd.exe -> node`. Killing only the
        wrapper leaves node alive holding the stdio pipes, which makes
        proc.wait() hang until it times out and leaks the process.
        """
        pid = proc.pid
        if pid is None:
            return
        if IS_WIN:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    async def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._resolve(msg)
        finally:
            # stdout closed: fail in-flight calls instead of waiting forever.
            self._fail_pending(McpError(f"MCP server '{self.server_name}' exited"))

    async def _request(self, method: str, params: dict | None = None) -> Any:
        if not self._process or not self._process.stdin:
            raise McpError(f"MCP server '{self.server_name}' is not running")

        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        # Register before writing: a fast local server may reply immediately.
        self._pending[req_id] = fut
        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        try:
            self._process.stdin.write((msg + "\n").encode())
            await self._process.stdin.drain()
        except Exception:
            self._pending.pop(req_id, None)
            fut.cancel()
            raise
        try:
            return await fut
        except asyncio.CancelledError:
            # Timed out or aborted mid-flight: drop the slot so a late reply
            # cannot resolve an abandoned future.
            self._pending.pop(req_id, None)
            fut.cancel()
            raise

    async def _notify(self, method: str, params: dict | None = None) -> None:
        if not self._process or not self._process.stdin:
            return
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}})
        try:
            self._process.stdin.write((msg + "\n").encode())
            await self._process.stdin.drain()
        except Exception:
            pass  # notifications are fire-and-forget

    def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        proc = self._process
        self._process = None
        if proc is not None:
            self._terminate(proc)
        super().close()

    async def aclose(self) -> None:
        proc = self._process
        self.close()
        if proc is not None:
            try:
                # Reap the child so it doesn't linger as a zombie or emit
                # "Event loop is closed" noise during interpreter shutdown.
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass


# ─── HTTP / SSE transport ────────────────────────────────────


class HttpMcpConnection(_McpConnection):
    """Remote server: Streamable HTTP, with legacy HTTP+SSE fallback.

    Streamable HTTP sends one POST per request; the reply comes back either as
    a plain JSON body or as a short SSE stream. Legacy HTTP+SSE keeps a
    long-lived GET stream open and POSTs requests to the endpoint it announces.
    """

    transport = "http"

    def __init__(
        self,
        server_name: str,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        transport: str = "auto",
        client: Any = None,
        **kwargs: Any,
    ):
        super().__init__(server_name, **kwargs)
        self.url = url
        self.headers = dict(headers or {})
        self.mode = transport if transport in ("http", "sse") else "auto"
        self._client = client
        self._owns_client = client is None
        self._session_id: str | None = None

        # Legacy HTTP+SSE state
        self._post_url: str | None = None
        self._reader_task: asyncio.Task | None = None
        self._stream_cm = None
        self._response = None
        self._endpoint_ready: asyncio.Event | None = None
        self._stream_error: BaseException | None = None

    # ── lifecycle ──

    async def connect(self) -> None:
        httpx = _require_httpx()
        if self._client is None:
            timeout = httpx.Timeout(self.tool_timeout, connect=self.connect_timeout)
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        if self.mode == "sse":
            await self._open_sse_stream()

    def _http_headers(self, *, body: bool, accept: str | None = None) -> dict[str, str]:
        headers = {"Accept": accept or "application/json, text/event-stream"}
        if body:
            headers["Content-Type"] = "application/json"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        headers.update(self.headers)
        return headers

    def _capture_session(self, response) -> None:
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

    # ── requests ──

    async def _request(self, method: str, params: dict | None = None) -> Any:
        if self._post_url is not None:  # legacy SSE after the handshake
            return await self._post_and_wait(method, params)
        try:
            return await self._post_streamable(method, params)
        except _HttpStatusError as exc:
            if self.mode == "auto" and exc.status in _SSE_FALLBACK_STATUS:
                # Not a Streamable HTTP endpoint — downgrade to legacy SSE.
                self.mode = "sse"
                await self._open_sse_stream()
                return await self._post_and_wait(method, params)
            raise

    async def _post_streamable(self, method: str, params: dict | None) -> Any:
        req_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}

        async with self._client.stream(
            "POST", self.url, json=payload, headers=self._http_headers(body=True),
        ) as response:
            self._capture_session(response)
            if response.status_code >= 400:
                await response.aread()
                raise _HttpStatusError(
                    response.status_code,
                    f"POST {self.url} returned HTTP {response.status_code}",
                )
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # Server chose to answer inside an SSE stream.
                async for _event, data in _iter_sse(response):
                    try:
                        msg = _loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and msg.get("id") == req_id:
                        return _unwrap(msg)
                raise McpError(
                    f"server '{self.server_name}' closed the SSE stream without replying to {method}"
                )
            body = await response.aread()

        if not body:
            return None  # e.g. 202 Accepted for a notification
        return _find_response(_loads(body), req_id)

    async def _post_and_wait(self, method: str, params: dict | None) -> Any:
        """Legacy SSE: POST the request, then wait for it on the open stream."""
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        try:
            response = await self._client.post(
                self._post_url, json=payload, headers=self._http_headers(body=True),
            )
            if response.status_code >= 400:
                raise _HttpStatusError(
                    response.status_code,
                    f"POST {self._post_url} returned HTTP {response.status_code}",
                )
        except Exception:
            self._pending.pop(req_id, None)
            fut.cancel()
            raise
        try:
            return await fut
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            fut.cancel()
            raise

    async def _notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        url = self._post_url or self.url
        try:
            response = await self._client.post(
                url, json=payload, headers=self._http_headers(body=True),
            )
            self._capture_session(response)
        except Exception:
            pass  # notifications are fire-and-forget

    # ── legacy HTTP+SSE ──

    async def _open_sse_stream(self) -> None:
        self.transport = "sse"
        self._endpoint_ready = asyncio.Event()
        cm = self._client.stream(
            "GET", self.url,
            headers=self._http_headers(body=False, accept="text/event-stream"),
        )
        self._response = await cm.__aenter__()
        self._stream_cm = cm
        self._reader_task = asyncio.create_task(self._sse_read_loop())

        try:
            await asyncio.wait_for(
                self._endpoint_ready.wait(), timeout=self.connect_timeout,
            )
        except asyncio.TimeoutError as exc:
            await self._teardown_stream()
            raise McpError(
                f"server '{self.server_name}' sent no 'endpoint' event within "
                f"{self.connect_timeout:g}s"
            ) from exc
        if self._post_url is None:
            error = self._stream_error
            await self._teardown_stream()
            raise McpError(
                f"server '{self.server_name}' closed the SSE stream during the handshake"
                + (f": {error}" if error else "")
            )

    async def _teardown_stream(self) -> None:
        """Drop the SSE stream and its reader without touching the HTTP client."""
        task = self._stop_reader()
        cm, self._stream_cm = self._stream_cm, None
        self._response = None
        if task is not None:
            try:
                await task
            except BaseException:
                pass
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    async def _sse_read_loop(self) -> None:
        assert self._response is not None
        try:
            async for event, data in _iter_sse(self._response):
                if event == "endpoint":
                    self._post_url = urljoin(self.url, data.strip())
                    assert self._endpoint_ready is not None
                    self._endpoint_ready.set()
                elif event == "message":
                    try:
                        msg = _loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict):
                        self._resolve(msg)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - surfaced via _stream_error
            self._stream_error = exc
        finally:
            if self._endpoint_ready is not None:
                self._endpoint_ready.set()
            self._fail_pending(
                self._stream_error
                or McpError(f"SSE stream from '{self.server_name}' closed")
            )

    # ── teardown ──

    def _stop_reader(self) -> asyncio.Task | None:
        """Cancel the SSE reader exactly once and hand it back for awaiting."""
        task, self._reader_task = self._reader_task, None
        if task is not None:
            task.cancel()
        return task

    def close(self) -> None:
        self._stop_reader()
        super().close()

    async def aclose(self) -> None:
        await self._teardown_stream()
        super().close()  # reject any in-flight requests
        if self._owns_client and self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None


# ─── MCP Manager ─────────────────────────────────────────────


def _validate_server_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError("name may only contain letters, digits, '_' and '-'")
    if "__" in name:
        raise ValueError("name must not contain '__' (it separates server and tool)")


class McpManager:
    """Manages all MCP server connections. Call load_and_connect() once, then
    use get_tool_definitions() and call_tool() to integrate with the agent."""

    def __init__(self):
        self._connections: dict[str, _McpConnection] = {}
        self._tools: list[dict] = []
        self._connected = False

    async def load_and_connect(self) -> bool:
        """Read config, connect to every configured server, discover tools."""
        if self._connected:
            return True

        configs = self._load_configs()
        if not configs:
            self._connected = True
            return True

        retry_needed = False
        for name, cfg in configs.items():
            if name in self._connections:
                continue
            try:
                conn = self._build_connection(name, cfg)
            except ValueError as e:
                print(f"[mcp] Skipping '{name}': {e}", flush=True)
                continue
            try:
                await conn.connect()
                await asyncio.wait_for(conn.initialize(), timeout=conn.connect_timeout)
                server_tools = await asyncio.wait_for(conn.list_tools(), timeout=conn.connect_timeout)
                self._connections[name] = conn
                self._tools.extend(server_tools)
                flags = " [read-only]" if conn.read_only else ""
                print(
                    f"[mcp] Connected to '{name}' via {conn.transport} — "
                    f"{len(server_tools)} tools{flags}",
                    flush=True,
                )
            except Exception as e:
                retry_needed = True
                print(f"[mcp] Failed to connect to '{name}': {e}", flush=True)
                await conn.aclose()

        # Leave transient failures retryable. Invalid configurations are
        # reported and skipped rather than retried on every user turn; already
        # connected servers are skipped above on a subsequent attempt.
        self._connected = not retry_needed
        return self._connected

    def _build_connection(self, name: str, cfg: dict) -> _McpConnection:
        _validate_server_name(name)
        options = {
            "connect_timeout": _as_float(cfg.get("connectTimeout"), DEFAULT_CONNECT_TIMEOUT),
            "tool_timeout": _as_float(cfg.get("toolTimeout"), DEFAULT_TOOL_TIMEOUT),
            "read_only": bool(cfg.get("readOnly", False)),
            "min_interval": _min_interval_from(cfg),
        }
        url = cfg.get("url")
        if url:
            url = str(url)
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"'url' must start with http:// or https:// (got {url!r})")
            return HttpMcpConnection(
                name, url, cfg.get("headers"),
                transport=str(cfg.get("transport", "auto")).lower(),
                **options,
            )
        return McpConnection(
            name, cfg["command"], cfg.get("args"), cfg.get("env"), **options,
        )

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions in Anthropic API format with the mcp__ prefix."""
        definitions: list[dict] = []
        for tool in self._tools:
            full_name = f"mcp__{tool['serverName']}__{tool['name']}"
            if not _NAME_RE.match(full_name) or len(full_name) > _MAX_PREFIXED_TOOL_LEN:
                print(
                    f"[mcp] Skipping tool '{full_name}': name must match "
                    f"{_NAME_RE.pattern} and be at most {_MAX_PREFIXED_TOOL_LEN} chars",
                    flush=True,
                )
                continue
            definitions.append({
                "name": full_name,
                "description": tool.get("description") or f"MCP tool {tool['name']} from {tool['serverName']}",
                "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
            })
        return definitions

    def is_mcp_tool(self, name: str) -> bool:
        """Check if a tool name is an MCP-prefixed tool."""
        return name.startswith("mcp__")

    def is_concurrency_safe(self, prefixed_name: str) -> bool:
        """Whether this MCP tool may run in parallel with other tool calls.

        Opt-in per server via "readOnly": true — an MCP tool can have arbitrary
        side effects, so running one concurrently with a sibling call is only
        safe when the operator says the server is read-only.
        """
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            return False
        conn = self._connections.get(parts[1])
        return bool(conn and conn.read_only)

    async def call_tool(self, prefixed_name: str, args: dict) -> str:
        """Route a prefixed tool call to the correct server.

        Failures come back as an error string rather than an exception: the
        agent loop feeds the returned text back to the model, so a dead or slow
        server degrades into a visible tool error instead of killing the turn.
        """
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            return f"[mcp] Invalid MCP tool name: {prefixed_name}"
        server_name = parts[1]
        tool_name = "__".join(parts[2:])  # tool names may themselves contain __

        conn = self._connections.get(server_name)
        if conn is None:
            return f"[mcp] Server '{server_name}' is not connected."

        try:
            return await conn.call_tool(tool_name, args)
        except McpTimeoutError as e:
            return (
                f"[mcp] {e}. The server may be slow or hung — raise 'toolTimeout' "
                f"for '{server_name}' in .mcp.json if it legitimately needs longer."
            )
        except McpError as e:
            return f"[mcp] {e}"
        except Exception as e:  # noqa: BLE001 - never break the agent loop
            return f"[mcp] Unexpected error calling '{prefixed_name}': {type(e).__name__}: {e}"

    async def disconnect_all(self) -> None:
        """Shut down every server. Safe to call more than once."""
        connections = list(self._connections.values())
        self._connections.clear()
        self._tools.clear()
        self._connected = False
        for conn in connections:
            try:
                await conn.aclose()
            except Exception:
                pass

    # ── Config loading ──

    def _load_configs(self) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        self._merge_config_file(Path.home() / ".claude" / "settings.json", merged)
        self._merge_config_file(Path.cwd() / ".claude" / "settings.json", merged)
        self._merge_config_file(Path.cwd() / ".mcp.json", merged)
        return merged

    def _merge_config_file(self, path: Path, target: dict[str, dict]) -> None:
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[mcp] Ignoring malformed config {path}: {e}", flush=True)
            return
        if not isinstance(raw, dict):
            return
        # A settings.json without an "mcpServers" section carries unrelated keys
        # (env, modelSettings, ...) — only .mcp.json is a bare server map, and
        # only in that case is a bad entry worth complaining about.
        explicit = raw.get("mcpServers")
        servers = explicit if isinstance(explicit, dict) else raw
        strict = isinstance(explicit, dict)
        for name, config in servers.items():
            if not isinstance(config, dict):
                continue
            if "url" not in config and "serverUrl" in config:
                config = {**config, "url": config["serverUrl"]}
            if "command" not in config and "url" not in config:
                if strict:
                    print(
                        f"[mcp] Ignoring '{name}' in {path.name}: "
                        f"needs a 'command' (stdio) or 'url' (HTTP) field",
                        flush=True,
                    )
                continue
            target[name] = config


def _min_interval_from(cfg: dict) -> float:
    """Seconds to leave between consecutive tool calls to one server.

    "minInterval" is the direct knob; "maxQps" is the friendlier spelling for
    services that document a requests-per-second cap. Zero means no throttling.
    """
    explicit = _as_float(cfg.get("minInterval"), 0.0)
    if explicit > 0:
        return explicit
    qps = _as_float(cfg.get("maxQps"), 0.0)
    return 1.0 / qps if qps > 0 else 0.0


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
