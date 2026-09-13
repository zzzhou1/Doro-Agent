from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mini_claude.agent import Agent


def test_openai_backend_does_not_require_custom_base_url() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI") as client:
        agent = Agent(
            backend="openai",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    assert agent.backend == "openai"
    assert agent.use_openai is True
    client.assert_called_once_with(base_url=None, api_key="test-key")


def test_anthropic_backend_preserves_custom_base_url() -> None:
    with patch("mini_claude.agent.anthropic.AsyncAnthropic") as client:
        agent = Agent(
            backend="anthropic",
            api_key="test-key",
            anthropic_base_url="https://anthropic.example.invalid",
            custom_system_prompt="test prompt",
        )
    assert agent.backend == "anthropic"
    assert agent.use_openai is False
    client.assert_called_once_with(
        api_key="test-key",
        base_url="https://anthropic.example.invalid",
    )


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        Agent(backend="unknown", custom_system_prompt="test prompt")


def test_switch_model_keeps_backend_and_refreshes_model_state() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            model="gpt-4o",
            api_key="test-key",
            thinking=True,
            custom_system_prompt="test prompt",
        )

    assert agent.switch_model("  custom-model  ") == "custom-model"
    assert agent.model == "custom-model"
    assert agent.backend == "openai"
    assert agent.effective_window == 180000
    assert agent._thinking_mode == "disabled"
    with pytest.raises(ValueError, match="empty"):
        agent.switch_model("   ")


@pytest.mark.asyncio
async def test_openai_model_list_includes_current_model() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            model="current-model",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    agent._openai_client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id="z-model"), {"id": "a-model"}]
        )
    )

    assert await agent.list_models() == ["a-model", "current-model", "z-model"]
    agent._openai_client.models.list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_anthropic_model_list_uses_bounded_request() -> None:
    with patch("mini_claude.agent.anthropic.AsyncAnthropic"):
        agent = Agent(
            backend="anthropic",
            model="claude-current",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    agent._anthropic_client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id="claude-other")])
    )

    assert await agent.list_models() == ["claude-current", "claude-other"]
    agent._anthropic_client.models.list.assert_awaited_once_with(limit=100)


@pytest.mark.asyncio
async def test_model_list_rejects_empty_provider_response() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    agent._openai_client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[])
    )

    with pytest.raises(RuntimeError, match="returned no models"):
        await agent.list_models()


def test_restore_session_restores_model_id_and_history() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            model="gpt-4o",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    messages = [
        {"role": "system", "content": "test prompt"},
        {"role": "user", "content": "hello"},
    ]

    agent.restore_session({
        "metadata": {
            "id": "saved123",
            "model": "gpt-4o-mini",
            "backend": "openai",
            "startTime": "2026-01-01T00:00:00Z",
            "preview": "hello",
        },
        "openaiMessages": messages,
        "anthropicMessages": None,
    })

    assert agent.session_id == "saved123"
    assert agent.model == "gpt-4o-mini"
    assert agent._openai_messages == messages
    assert agent.has_conversation_history() is True


def test_restore_session_rejects_cross_backend_history() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )

    with pytest.raises(ValueError, match="Cross-backend"):
        agent.restore_session({
            "metadata": {"id": "saved123", "backend": "anthropic"},
            "anthropicMessages": [{"role": "user", "content": "hello"}],
        })


def test_auto_save_records_resumable_session_metadata() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            model="gpt-4o-mini",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    agent._openai_messages.append({"role": "user", "content": "hello"})
    agent._last_user_preview = "hello"

    with patch("mini_claude.agent.save_session") as save:
        agent._auto_save()

    saved_id, payload = save.call_args.args
    assert saved_id == agent.session_id
    assert payload["metadata"]["backend"] == "openai"
    assert payload["metadata"]["model"] == "gpt-4o-mini"
    assert payload["metadata"]["preview"] == "hello"
    assert payload["metadata"]["schemaVersion"] == 2
    assert payload["metadata"]["updatedAt"]


@pytest.mark.asyncio
async def test_mcp_initialization_retries_and_deduplicates_tools() -> None:
    with patch("mini_claude.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    agent.tools = []
    tool = {
        "name": "mcp__demo__lookup",
        "description": "lookup",
        "input_schema": {"type": "object", "properties": {}},
    }
    manager = SimpleNamespace(
        load_and_connect=AsyncMock(side_effect=[False, True]),
        get_tool_definitions=Mock(side_effect=[[], [tool]]),
    )
    agent._mcp_manager = manager

    await agent._ensure_mcp_initialized()
    assert agent._mcp_initialized is False
    assert agent.tools == []

    await agent._ensure_mcp_initialized()
    assert agent._mcp_initialized is True
    assert [item["name"] for item in agent.tools] == ["mcp__demo__lookup"]

    # Once initialized, later calls are cheap and cannot duplicate definitions.
    await agent._ensure_mcp_initialized()
    assert manager.load_and_connect.await_count == 2
    assert [item["name"] for item in agent.tools] == ["mcp__demo__lookup"]
