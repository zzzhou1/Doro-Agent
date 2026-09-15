from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from doro.agent import Agent
from doro.reasoning import reasoning_fallback_for


def test_openai_backend_does_not_require_custom_base_url() -> None:
    with patch("doro.agent.openai.AsyncOpenAI") as client:
        agent = Agent(
            backend="openai",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    assert agent.backend == "openai"
    assert agent.use_openai is True
    client.assert_called_once_with(base_url=None, api_key="test-key")


def test_anthropic_backend_preserves_custom_base_url() -> None:
    with patch("doro.agent.anthropic.AsyncAnthropic") as client:
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
    with patch("doro.agent.openai.AsyncOpenAI"):
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
    assert agent.effective_window == 252000
    assert agent.reasoning_effort == "high"
    assert agent._thinking_mode == "reasoning"
    with pytest.raises(ValueError, match="empty"):
        agent.switch_model("   ")


@pytest.mark.asyncio
async def test_openai_model_list_includes_current_model() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
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
    with patch("doro.agent.anthropic.AsyncAnthropic"):
        agent = Agent(
            backend="anthropic",
            model="claude-current",
            api_key="test-key",
            reasoning_effort="off",
            custom_system_prompt="test prompt",
        )
    agent._anthropic_client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id="claude-other")])
    )

    assert await agent.list_models() == ["claude-current", "claude-other"]
    agent._anthropic_client.models.list.assert_awaited_once_with(limit=100)


@pytest.mark.asyncio
async def test_model_list_rejects_empty_provider_response() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
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
    with patch("doro.agent.openai.AsyncOpenAI"):
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
            "reasoningEffort": "low",
        },
        "openaiMessages": messages,
        "anthropicMessages": None,
    })

    assert agent.session_id == "saved123"
    assert agent.model == "gpt-4o-mini"
    assert agent._openai_messages == messages
    assert agent.has_conversation_history() is True
    assert agent.reasoning_effort == "low"


def test_restore_session_rejects_cross_backend_history() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
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
    with patch("doro.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            model="gpt-4o-mini",
            api_key="test-key",
            custom_system_prompt="test prompt",
        )
    agent._openai_messages.append({"role": "user", "content": "hello"})
    agent._last_user_preview = "hello"

    with patch("doro.agent.save_session") as save:
        agent._auto_save()

    saved_id, payload = save.call_args.args
    assert saved_id == agent.session_id
    assert payload["metadata"]["backend"] == "openai"
    assert payload["metadata"]["model"] == "gpt-4o-mini"
    assert payload["metadata"]["preview"] == "hello"
    assert payload["metadata"]["reasoningEffort"] == "medium"
    assert payload["metadata"]["schemaVersion"] == 3
    assert payload["metadata"]["updatedAt"]


def test_default_effort_and_legacy_thinking_alias() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
        default_agent = Agent(backend="openai", api_key="test-key")
        thinking_agent = Agent(backend="openai", api_key="test-key", thinking=True)

    assert default_agent.reasoning_effort == "medium"
    assert default_agent._openai_reasoning_params() == {"reasoning_effort": "medium"}
    assert thinking_agent.reasoning_effort == "high"


def test_openai_effort_wire_mapping_and_reset() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            api_key="test-key",
            reasoning_effort="auto",
            default_reasoning_effort="low",
        )

    assert agent._openai_reasoning_params() == {}
    assert agent.set_reasoning_effort("off") == "off"
    assert agent._openai_reasoning_params() == {"reasoning_effort": "none"}
    assert agent.reset_reasoning_effort() == "low"
    assert agent._reasoning_effort_explicit is False


def test_anthropic_adaptive_effort_mapping() -> None:
    with patch("doro.agent.anthropic.AsyncAnthropic"):
        agent = Agent(
            backend="anthropic",
            model="claude-opus-5",
            api_key="test-key",
            reasoning_effort="high",
        )

    assert agent._anthropic_reasoning_params() == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    agent.set_reasoning_effort("off")
    assert agent._anthropic_reasoning_params() == {
        "thinking": {"type": "disabled"}
    }


def test_anthropic_never_uses_legacy_thinking_budget() -> None:
    with patch("doro.agent.anthropic.AsyncAnthropic"):
        agent = Agent(
            backend="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            reasoning_effort="low",
        )

    assert agent._thinking_mode == "adaptive"
    assert agent._anthropic_reasoning_params() == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }
    assert "budget_tokens" not in str(agent._anthropic_reasoning_params())


def test_minimal_is_invalid_and_model_switch_defers_capability_check() -> None:
    class ApiError(Exception):
        def __init__(self, status_code: int, message: str):
            super().__init__(message)
            self.status_code = status_code

    with patch("doro.agent.anthropic.AsyncAnthropic"):
        with pytest.raises(ValueError, match="Unknown reasoning effort"):
            Agent(
                backend="anthropic",
                model="claude-opus-5",
                api_key="test-key",
                reasoning_effort="minimal",
            )
        agent = Agent(
            backend="anthropic",
            model="claude-opus-5",
            api_key="test-key",
            reasoning_effort="high",
        )

    assert agent.switch_model("claude-3-5-sonnet") == "claude-3-5-sonnet"
    assert agent.model == "claude-3-5-sonnet"
    assert agent.reasoning_effort == "high"
    assert agent._thinking_mode == "adaptive"
    assert reasoning_fallback_for(
        ApiError(400, "unknown parameter: reasoning_effort"), "high"
    ) == "auto"
    assert reasoning_fallback_for(
        ApiError(422, "unsupported value for reasoning_effort"), "high"
    ) == "medium"
    assert reasoning_fallback_for(
        ApiError(401, "unsupported value for reasoning_effort"), "high"
    ) is None
    assert reasoning_fallback_for(
        ApiError(400, "context length is invalid"), "high"
    ) is None
    assert reasoning_fallback_for(
        ApiError(400, "unknown parameter: reasoning_effort"), "off"
    ) is None


def test_explicit_effort_beats_restored_session_effort() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            api_key="test-key",
            reasoning_effort="high",
            reasoning_effort_explicit=True,
        )

    agent.restore_session({
        "metadata": {
            "id": "saved123",
            "backend": "openai",
            "reasoningEffort": "low",
        },
        "openaiMessages": [{"role": "system", "content": "test"}],
    })

    assert agent.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_openai_stream_uses_visible_bounded_reasoning_fallback() -> None:
    class CompatibilityError(Exception):
        status_code = 400

    class EmptyStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with patch("doro.agent.openai.AsyncOpenAI"):
        agent = Agent(
            backend="openai",
            api_key="test-key",
            reasoning_effort="xhigh",
            custom_tools=[],
        )
    agent._openai_client.chat.completions.create = AsyncMock(
        side_effect=[
            CompatibilityError("unsupported value for reasoning_effort"),
            EmptyStream(),
        ]
    )

    with patch("doro.agent.emit_warning") as warned:
        await agent._call_openai_stream()

    calls = agent._openai_client.chat.completions.create.await_args_list
    assert calls[0].kwargs["reasoning_effort"] == "xhigh"
    assert calls[1].kwargs["reasoning_effort"] == "high"
    assert agent.reasoning_effort == "high"
    warned.assert_called_once()

    async def fails_after_output(state: dict[str, bool]):
        state["started"] = True
        raise CompatibilityError("unsupported value for reasoning_effort")

    with pytest.raises(CompatibilityError):
        await agent._with_reasoning_fallback(fails_after_output)
    assert agent.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_mcp_initialization_retries_and_deduplicates_tools() -> None:
    with patch("doro.agent.openai.AsyncOpenAI"):
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
        wait_ready=AsyncMock(side_effect=[False, True]),
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
    assert manager.wait_ready.await_count == 2
    assert [item["name"] for item in agent.tools] == ["mcp__demo__lookup"]
