from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mini_claude.__main__ import (
    _load_project_env,
    _resolve_model_selector,
    _resolve_api_config,
    _resolve_session_selector,
    _restore_agent_session,
    run_repl,
)


ROOT = Path(__file__).resolve().parents[1]


def _args(api_base: str | None = None) -> Namespace:
    return Namespace(api_base=api_base)


def test_console_script_metadata_matches_documentation() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"] == {
        "mini-claude": "mini_claude.__main__:main"
    }
    assert metadata["project"]["requires-python"] == ">=3.11"


def test_module_help_runs_without_api_key() -> None:
    env = os.environ.copy()
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-m", "mini_claude", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage: mini-claude" in result.stdout


def test_openai_key_uses_official_default_endpoint() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=True):
        assert _resolve_api_config(_args()) == ("openai", "test-openai-key", None)


def test_openai_compatible_endpoint_is_preserved() -> None:
    env = {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_BASE_URL": "https://example.invalid/v1",
    }
    with patch.dict(os.environ, env, clear=True):
        assert _resolve_api_config(_args()) == (
            "openai",
            "test-openai-key",
            "https://example.invalid/v1",
        )


def test_anthropic_is_preferred_over_openai_without_base_url() -> None:
    env = {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "ANTHROPIC_BASE_URL": "https://anthropic.example.invalid",
        "OPENAI_API_KEY": "test-openai-key",
    }
    with patch.dict(os.environ, env, clear=True):
        assert _resolve_api_config(_args()) == (
            "anthropic",
            "test-anthropic-key",
            "https://anthropic.example.invalid",
        )


def test_explicit_api_base_selects_openai_compatible_backend() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "shared-key"}, clear=True):
        assert _resolve_api_config(_args("https://provider.example/v1")) == (
            "openai",
            "shared-key",
            "https://provider.example/v1",
        )


def test_project_dotenv_loads_without_overriding_shell(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n"
        "OPENAI_BASE_URL=https://dotenv.example/v1\n"
        "MINI_CLAUDE_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {"OPENAI_API_KEY": "from-shell"}, clear=True):
        assert _load_project_env(tmp_path) is True
        assert os.environ["OPENAI_API_KEY"] == "from-shell"
        assert os.environ["OPENAI_BASE_URL"] == "https://dotenv.example/v1"
        assert os.environ["MINI_CLAUDE_MODEL"] == "dotenv-model"


def test_session_selector_accepts_list_number_or_id() -> None:
    sessions = [{"id": "abcd1234"}, {"id": "12345678"}]

    assert _resolve_session_selector("1", sessions) == "abcd1234"
    assert _resolve_session_selector("12345678", sessions) == "12345678"
    with pytest.raises(ValueError, match="out of range"):
        _resolve_session_selector("3", sessions)
    with pytest.raises(ValueError, match="not found"):
        _resolve_session_selector("missing", sessions)


def test_model_selector_accepts_list_number_or_exact_name() -> None:
    models = ["alpha", "12345678", "zeta"]

    assert _resolve_model_selector("1", models) == "alpha"
    assert _resolve_model_selector("12345678", models) == "12345678"
    with pytest.raises(ValueError, match="out of range"):
        _resolve_model_selector("4", models)
    with pytest.raises(ValueError, match="not in the returned list"):
        _resolve_model_selector("missing", models)


@pytest.mark.asyncio
async def test_repl_model_selection_switches_by_number(monkeypatch) -> None:
    agent = Mock()
    agent.backend = "openai"
    agent.model = "alpha"
    agent._aborted = False
    agent._output_buffer = None
    agent.list_models = AsyncMock(return_value=["alpha", "beta"])
    agent.switch_model.return_value = "beta"
    responses = iter(["/model", "2", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    with (
        patch("mini_claude.__main__.signal.signal"),
        patch("mini_claude.__main__.print_welcome"),
        patch("mini_claude.__main__.print_user_prompt"),
        patch("mini_claude.__main__.print_info"),
    ):
        await run_repl(agent)

    agent.list_models.assert_awaited_once_with()
    agent.switch_model.assert_called_once_with("beta")


def test_restore_helper_rejects_other_project(tmp_path: Path, monkeypatch) -> None:
    agent = Mock()
    monkeypatch.chdir(tmp_path)
    with patch("mini_claude.__main__.load_session", return_value={
        "metadata": {
            "id": "saved123",
            "cwd": str(tmp_path / "another-project"),
        }
    }):
        with pytest.raises(ValueError, match="different project"):
            _restore_agent_session(agent, "saved123")
    agent.restore_session.assert_not_called()
