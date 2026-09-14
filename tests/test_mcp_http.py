"""HTTP / SSE transport tests — driven by httpx.MockTransport, no real sockets."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from doro.mcp_client import (
    _MAX_PREFIXED_TOOL_LEN,
    HttpMcpConnection,
    McpError,
)

TOOLS = [
    {
        "name": "ping",
        "description": "Ping the server",
        "inputSchema": {"type": "object", "properties": {}},
    }
]

ENDPOINT = "https://host.test/mcp"
MESSAGES = "/messages?sessionId=abc"


def _result(req_id: int, result: object) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _dispatch(payload: dict) -> dict | None:
    """Answer the three methods this test client actually uses."""
    method = payload["method"]
    if method == "initialize":
        return _result(payload["id"], {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock", "version": "1.0.0"},
        })
    if method == "tools/list":
        return _result(payload["id"], {"tools": TOOLS})
    if method == "tools/call":
        args = payload["params"]["arguments"]
        text = f"pong:{json.dumps(args, sort_keys=True)}"
        return _result(payload["id"], {"content": [{"type": "text", "text": text}]})
    return {
        "jsonrpc": "2.0",
        "id": payload["id"],
        "error": {"code": -32601, "message": f"unknown method {method}"},
    }


def _json_handler(seen: list[httpx.Request]):
    """A Streamable HTTP server that answers each POST with a JSON body."""

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        payload = json.loads(request.content)
        if "id" not in payload:  # notification
            return httpx.Response(202)
        return httpx.Response(
            200,
            json=_dispatch(payload),
            headers={"content-type": "application/json", "mcp-session-id": "sess-1"},
        )

    return handler


async def _open(handler, **kwargs) -> tuple[HttpMcpConnection, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    conn = HttpMcpConnection("mock", ENDPOINT, client=client, **kwargs)
    await conn.connect()
    return conn, client


@pytest.mark.asyncio
async def test_streamable_http_json_round_trip() -> None:
    seen: list[httpx.Request] = []
    conn, client = await _open(_json_handler(seen), headers={"Authorization": "Bearer secret"})
    try:
        await asyncio.wait_for(conn.initialize(), timeout=5)
        tools = await asyncio.wait_for(conn.list_tools(), timeout=5)
        assert [t["name"] for t in tools] == ["ping"]
        assert conn.transport == "http"

        result = await asyncio.wait_for(conn.call_tool("ping", {"q": 1}), timeout=5)
        assert result == 'pong:{"q": 1}'

        # The server handed us a session id; every later request must carry it.
        assert all(r.headers["authorization"] == "Bearer secret" for r in seen)
        assert seen[-1].headers.get("mcp-session-id") == "sess-1"
    finally:
        await conn.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_streamable_http_response_inside_an_sse_stream() -> None:
    """Some servers answer a POST with text/event-stream instead of JSON."""

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        msg = _dispatch(payload)
        body = f": keep-alive\n\nevent: message\ndata: {json.dumps(msg)}\n\n"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )

    conn, client = await _open(handler)
    try:
        assert await asyncio.wait_for(conn.call_tool("ping", {}), timeout=5) == "pong:{}"
    finally:
        await conn.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_http_error_becomes_a_tool_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    conn, client = await _open(handler)
    try:
        with pytest.raises(McpError, match="HTTP 500"):
            await asyncio.wait_for(conn.call_tool("ping", {}), timeout=5)
    finally:
        await conn.aclose()
        await client.aclose()


class _LiveSSE(httpx.AsyncByteStream):
    """A stream the handler can push events into once the client has connected."""

    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    async def __aiter__(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_legacy_http_sse_fallback() -> None:
    """A 405 on POST means 'not Streamable HTTP' — fall back to HTTP+SSE."""
    queue: asyncio.Queue = asyncio.Queue()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            await queue.put(b"event: endpoint\ndata: " + MESSAGES.encode() + b"\n\n")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_LiveSSE(queue),
            )
        if request.url.path == "/mcp":
            return httpx.Response(405, text="Method Not Allowed")

        payload = json.loads(request.content)
        if "id" in payload:
            msg = _dispatch(payload)
            await queue.put(f"event: message\ndata: {json.dumps(msg)}\n\n".encode())
        return httpx.Response(202)

    conn, client = await _open(handler, transport="auto")
    try:
        await asyncio.wait_for(conn.initialize(), timeout=5)
        assert conn.transport == "sse"
        assert conn._post_url == "https://host.test" + MESSAGES

        tools = await asyncio.wait_for(conn.list_tools(), timeout=5)
        assert [t["name"] for t in tools] == ["ping"]
        assert await asyncio.wait_for(conn.call_tool("ping", {"q": 2}), timeout=5) == 'pong:{"q": 2}'
    finally:
        await conn.aclose()
        await client.aclose()

    # The first POST probed /mcp and was rejected; every later POST went to the
    # endpoint the server announced over the stream.
    posts = [r for r in seen if r.method == "POST"]
    assert posts[0].url.path == "/mcp"
    assert all(r.url.path == "/messages" for r in posts[1:])


@pytest.mark.asyncio
async def test_forced_sse_transport_skips_the_probe() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            await queue.put(b"event: endpoint\ndata: " + MESSAGES.encode() + b"\n\n")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_LiveSSE(queue),
            )
        payload = json.loads(request.content)
        if "id" in payload:
            await queue.put(f"event: message\ndata: {json.dumps(_dispatch(payload))}\n\n".encode())
        return httpx.Response(202)

    conn, client = await _open(handler, transport="sse")
    try:
        await asyncio.wait_for(conn.initialize(), timeout=5)
        assert seen[0].method == "GET"  # no probe POST at all
        assert await asyncio.wait_for(conn.call_tool("ping", {}), timeout=5) == "pong:{}"
    finally:
        await conn.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_concurrent_http_calls_do_not_cross_talk() -> None:
    seen: list[httpx.Request] = []
    conn, client = await _open(_json_handler(seen))
    try:
        payloads = [f"token-{i}" for i in range(12)]
        results = await asyncio.wait_for(
            asyncio.gather(*(conn.call_tool("ping", {"q": p}) for p in payloads)),
            timeout=10,
        )
        assert results == [f'pong:{{"q": "{p}"}}' for p in payloads]
    finally:
        await conn.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_prefix_and_length_limits_match_the_model_api() -> None:
    assert len("mcp__mock__ping") <= _MAX_PREFIXED_TOOL_LEN
