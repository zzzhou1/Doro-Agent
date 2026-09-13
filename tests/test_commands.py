"""Tests for the unified command registry (mini_claude.commands).

The registry exists because /effort once shipped in dispatch while missing
from the welcome banner — three hand-maintained lists had drifted apart.
These tests pin the four derived surfaces (welcome, completion, --help,
dispatch) to the single registry so they cannot drift again.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mini_claude.__main__ import _REPL_HANDLERS, run_repl, validate_handlers
from mini_claude.commands import (
    COMMAND_REGISTRY,
    SKILL_HELP_ENTRY,
    completion_map,
    find_command,
    format_help_section,
    help_entries,
    welcome_command_line,
)


def test_every_command_has_exactly_one_handler() -> None:
    validate_handlers()  # must not raise
    assert set(_REPL_HANDLERS) == {spec.name for spec in COMMAND_REGISTRY}


def test_every_command_appears_in_welcome_completion_and_help() -> None:
    welcome = welcome_command_line()
    completions = completion_map()
    help_text = format_help_section()
    for spec in COMMAND_REGISTRY:
        assert f"/{spec.name}" in welcome, spec.name
        assert completions[f"/{spec.name}"] == spec.description
        for usage, _ in spec.help_entries:
            assert usage in help_text, usage


def test_welcome_lists_effort_command() -> None:
    """Regression: /effort was missing from the hand-written welcome list."""
    assert "/effort" in welcome_command_line()


def test_help_section_includes_skill_invocation_form() -> None:
    usage, _ = SKILL_HELP_ENTRY
    assert usage in format_help_section()


def test_help_entries_align_with_registry_order() -> None:
    entries = help_entries()
    expected = [
        entry for spec in COMMAND_REGISTRY for entry in spec.help_entries
    ]
    assert entries == expected


def test_find_command_roundtrip() -> None:
    assert find_command("effort").takes_args is True
    assert find_command("clear").takes_args is False
    assert find_command("nope") is None


def _mock_agent() -> Mock:
    agent = Mock()
    agent.backend = "openai"
    agent._aborted = False
    agent._output_buffer = None
    return agent


@pytest.mark.asyncio
async def test_dispatch_runs_registered_handler() -> None:
    agent = _mock_agent()
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/clear", "exit"])

    with (
        patch("mini_claude.__main__.signal.signal"),
        patch("mini_claude.__main__.print_welcome"),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    agent.clear_history.assert_called_once_with()


@pytest.mark.asyncio
async def test_dispatch_passes_args_to_handler() -> None:
    agent = _mock_agent()
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/effort high", "exit"])

    with (
        patch("mini_claude.__main__.signal.signal"),
        patch("mini_claude.__main__.print_welcome"),
        patch("mini_claude.__main__.print_info"),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    agent.set_reasoning_effort.assert_called_once_with("high")


@pytest.mark.asyncio
async def test_args_on_no_arg_command_fall_through_to_chat() -> None:
    """`/clear please` must not clear history (pre-registry behavior)."""
    agent = _mock_agent()
    agent.chat = AsyncMock()
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/clear please", "exit"])

    with (
        patch("mini_claude.__main__.signal.signal"),
        patch("mini_claude.__main__.print_welcome"),
        patch("mini_claude.__main__.get_skill_by_name", return_value=None),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    agent.clear_history.assert_not_called()
    agent.chat.assert_awaited_once_with("/clear please")


@pytest.mark.asyncio
async def test_unknown_command_falls_through_to_chat() -> None:
    agent = _mock_agent()
    agent.chat = AsyncMock()
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/nope", "exit"])

    with (
        patch("mini_claude.__main__.signal.signal"),
        patch("mini_claude.__main__.print_welcome"),
        patch("mini_claude.__main__.get_skill_by_name", return_value=None),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    agent.chat.assert_awaited_once_with("/nope")
