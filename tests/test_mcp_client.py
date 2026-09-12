from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from mini_claude.mcp_client import McpConnection


SERVER = Path(__file__).parent / "fixtures" / "mcp_server.py"


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
