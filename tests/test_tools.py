from __future__ import annotations

import os
from pathlib import Path

import pytest

from mini_claude.tools import execute_tool


@pytest.mark.asyncio
async def test_existing_file_must_be_read_before_edit(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    state: dict[str, float] = {}

    denied = await execute_tool(
        "edit_file",
        {"file_path": str(target), "old_string": "value = 1", "new_string": "value = 2"},
        state,
    )
    assert "must read this file" in denied

    read_result = await execute_tool("read_file", {"file_path": str(target)}, state)
    assert "value = 1" in read_result

    edited = await execute_tool(
        "edit_file",
        {"file_path": str(target), "old_string": "value = 1", "new_string": "value = 2"},
        state,
    )
    assert edited.startswith("Successfully edited")
    assert target.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_external_change_invalidates_read_state(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    state: dict[str, float] = {}
    await execute_tool("read_file", {"file_path": str(target)}, state)

    previous_mtime = target.stat().st_mtime_ns
    target.write_text("value = 3\n", encoding="utf-8")
    os.utime(target, ns=(previous_mtime + 1_000_000_000, previous_mtime + 1_000_000_000))

    result = await execute_tool(
        "edit_file",
        {"file_path": str(target), "old_string": "value = 1", "new_string": "value = 2"},
        state,
    )
    assert "modified externally" in result
    assert target.read_text(encoding="utf-8") == "value = 3\n"


@pytest.mark.asyncio
async def test_quote_normalization_edit(tmp_path: Path) -> None:
    target = tmp_path / "quotes.py"
    target.write_text('message = “hello”\n', encoding="utf-8")
    state: dict[str, float] = {}
    await execute_tool("read_file", {"file_path": str(target)}, state)

    result = await execute_tool(
        "edit_file",
        {
            "file_path": str(target),
            "old_string": 'message = "hello"',
            "new_string": 'message = "world"',
        },
        state,
    )
    assert "matched via quote normalization" in result
    assert target.read_text(encoding="utf-8") == 'message = "world"\n'
