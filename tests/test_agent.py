from __future__ import annotations

from unittest.mock import patch

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
