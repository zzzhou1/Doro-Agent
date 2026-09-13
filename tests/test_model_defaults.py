"""One default model per backend.

A single hardcoded default cannot serve both backends: ``claude-opus-5`` means
nothing to an OpenAI-compatible gateway and ``gpt-5.6-sol`` means nothing to
Anthropic. The old CLI also *guessed* "the user did not choose a model" by
comparing the resolved name against that hardcoded default, so an explicit
``--model claude-opus-4-6`` on the OpenAI backend was silently rewritten to
``gpt-5.4``. These tests pin the replacement: resolve the backend first, then
fall back to that backend's own default.
"""

from __future__ import annotations

import os
from argparse import Namespace
from unittest.mock import patch

from mini_claude.agent import (
    DEFAULT_MODELS,
    Agent,
    default_model_for,
    resolve_default_model,
)
from mini_claude.__main__ import _resolve_model


def _args(model: str | None = None) -> Namespace:
    return Namespace(model=model)


def _model_env():
    """Environment with the model env vars blanked, everything else intact.

    Clearing the whole environment breaks both ``Path.home()`` (Agent() resolves
    its memory directory during construction) and the SDK clients, which refuse
    to be built without *some* key — so placeholders are supplied instead.
    """
    env = {
        "MINI_CLAUDE_MODEL": "",
        "ANTHROPIC_MODEL": "",
        "OPENAI_MODEL": "",
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
    }
    return patch.dict(os.environ, env, clear=False)


def test_each_backend_has_a_distinct_default() -> None:
    assert default_model_for("anthropic") == "claude-opus-5"
    assert default_model_for("openai") == "gpt-5.6-sol"
    assert DEFAULT_MODELS["anthropic"] != DEFAULT_MODELS["openai"]


def test_unknown_backend_falls_back_to_the_anthropic_default() -> None:
    assert default_model_for("some-future-backend") == DEFAULT_MODELS["anthropic"]


def test_backend_env_var_overrides_only_that_backend() -> None:
    with patch.dict(os.environ, {"OPENAI_MODEL": "oai-env"}, clear=True):
        assert resolve_default_model("openai") == "oai-env"
        assert resolve_default_model("anthropic") == "claude-opus-5"


def test_explicit_override_beats_the_backend_env_var() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_MODEL": "env-model"}, clear=True):
        assert resolve_default_model("anthropic", "explicit-model") == "explicit-model"


def test_global_env_var_does_not_leak_into_the_other_backend() -> None:
    """MINI_CLAUDE_MODEL is a CLI-level override, not a per-backend default."""
    with patch.dict(os.environ, {"MINI_CLAUDE_MODEL": "gpt-5.6-sol"}, clear=True):
        assert resolve_default_model("anthropic") == "claude-opus-5"


def test_blank_env_values_are_ignored() -> None:
    env = {"OPENAI_MODEL": "   ", "MINI_CLAUDE_MODEL": ""}
    with patch.dict(os.environ, env, clear=True):
        assert _resolve_model(_args(), "openai") == ("gpt-5.6-sol", "default for openai")


def test_cli_precedence() -> None:
    both = {"MINI_CLAUDE_MODEL": "global-model", "OPENAI_MODEL": "backend-model"}
    with patch.dict(os.environ, both, clear=True):
        assert _resolve_model(_args("cli-model"), "openai") == ("cli-model", "--model")
    with patch.dict(os.environ, both, clear=True):
        assert _resolve_model(_args(), "openai") == ("global-model", "MINI_CLAUDE_MODEL")
    with patch.dict(os.environ, {"OPENAI_MODEL": "backend-model"}, clear=True):
        assert _resolve_model(_args(), "openai") == ("backend-model", "OPENAI_MODEL")
    with _model_env():
        assert _resolve_model(_args(), "anthropic") == ("claude-opus-5", "default for anthropic")


def test_explicit_claude_model_survives_on_the_openai_backend() -> None:
    """Regression: the old code rewrote an explicit Claude --model to gpt-5.4."""
    with _model_env():
        assert _resolve_model(_args("claude-opus-4-6"), "openai") == (
            "claude-opus-4-6",
            "--model",
        )


def test_agent_resolves_its_own_default_when_model_is_omitted() -> None:
    with _model_env():
        assert Agent(backend="anthropic").model == "claude-opus-5"
        assert Agent(backend="openai").model == "gpt-5.6-sol"
        # api_base alone implies the OpenAI backend.
        assert Agent(api_base="https://gateway.example/v1").model == "gpt-5.6-sol"


def test_agent_keeps_an_explicit_model() -> None:
    with _model_env():
        assert Agent(backend="openai", model="my-model").model == "my-model"


def test_context_window_tracks_the_resolved_model() -> None:
    """effective_window keys off self.model, so it must not read a stale local."""
    with _model_env():
        openai_agent = Agent(backend="openai")
        anthropic_agent = Agent(backend="anthropic")
        assert openai_agent.context_window == 128000
        assert openai_agent.effective_window == 128000 - 20000
        assert anthropic_agent.context_window == 200000
        assert anthropic_agent.effective_window == 200000 - 20000


def test_status_snapshot_contains_display_state_but_no_credentials() -> None:
    with _model_env():
        agent = Agent(backend="openai", api_key="super-secret")
    snapshot = agent.get_status_snapshot()

    assert snapshot["backend"] == "openai"
    assert snapshot["model"] == "gpt-5.6-sol"
    assert snapshot["context_window"] == 128000
    assert snapshot["session_id"] == agent.session_id
    assert snapshot["reasoning_effort"] == "medium"
    assert "api_key" not in snapshot
    assert "api_base" not in snapshot
