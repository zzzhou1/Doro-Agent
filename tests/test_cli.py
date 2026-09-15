from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from doro.__main__ import (
    _force_utf8_stdio,
    _load_project_env,
    _resolve_api_config,
    _resolve_model_selector,
    _resolve_reasoning_effort,
    _resolve_session_selector,
    _restore_agent_session,
    parse_args,
    run_repl,
)

ROOT = Path(__file__).resolve().parents[1]


def _args(api_base: str | None = None) -> Namespace:
    return Namespace(api_base=api_base)


def test_force_utf8_stdio_tolerates_streams_without_reconfigure() -> None:
    """pytest's capture object, pipes, and mocks have no reconfigure()."""
    with patch.object(sys, "stdout", Mock(spec=[])), patch.object(sys, "stderr", Mock(spec=[])):
        _force_utf8_stdio()  # must not raise


def test_redirected_output_is_utf8_without_utf8_mode(tmp_path) -> None:
    """Redirecting stdout must not fall back to the locale code page.

    Windows only uses UTF-8 for the console; a pipe or file gets the ANSI code
    page (cp936 here), which garbles every non-ASCII character. The sandbox sets
    PYTHONUTF8/PYTHONIOENCODING, so they are removed to exercise the real path.
    """
    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    code = (
        "from doro.__main__ import _force_utf8_stdio;"
        "_force_utf8_stdio();"
        "import sys; sys.stdout.write('你好')"
    )
    out = tmp_path / "out.bin"
    with open(out, "wb") as fh:
        subprocess.run(
            [sys.executable, "-c", code],
            stdout=fh, stderr=subprocess.DEVNULL, env=env, cwd=str(ROOT), check=True,
        )

    assert out.read_bytes() == "你好".encode()


def test_console_script_metadata_matches_documentation() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"] == {
        "doro": "doro.__main__:main"
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
        [sys.executable, "-m", "doro", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage: doro" in result.stdout


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
        "DORO_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    with patch.dict(os.environ, {"OPENAI_API_KEY": "from-shell"}, clear=True):
        assert _load_project_env(tmp_path) is True
        assert os.environ["OPENAI_API_KEY"] == "from-shell"
        assert os.environ["OPENAI_BASE_URL"] == "https://dotenv.example/v1"
        assert os.environ["DORO_MODEL"] == "dotenv-model"


def _fake_package(tmp_path: Path, dotenv: str | None) -> Path:
    """Build a throwaway source tree and return its fake __main__.py path."""
    package_root = tmp_path / "proj"
    (package_root / "doro").mkdir(parents=True)
    fake_main = package_root / "doro" / "__main__.py"
    fake_main.write_text("", encoding="utf-8")
    if dotenv is not None:
        (package_root / ".env").write_text(dotenv, encoding="utf-8")
    return fake_main


def test_dotenv_falls_back_to_package_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doro.__main__ as cli

    fake_main = _fake_package(tmp_path, "ANTHROPIC_API_KEY=from-source-tree\n")
    (tmp_path / "work").mkdir()

    monkeypatch.setattr(cli, "__file__", str(fake_main))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    monkeypatch.chdir(tmp_path / "work")
    with patch.dict(os.environ, {}, clear=True):
        assert _load_project_env() is True
        assert os.environ["ANTHROPIC_API_KEY"] == "from-source-tree"


def test_dotenv_falls_back_to_user_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doro.__main__ as cli

    fake_main = _fake_package(tmp_path, None)
    home = tmp_path / "home"
    (home / ".doro").mkdir(parents=True)
    (home / ".doro" / ".env").write_text(
        "ANTHROPIC_API_KEY=from-user-dir\n", encoding="utf-8"
    )
    (tmp_path / "work").mkdir()

    monkeypatch.setattr(cli, "__file__", str(fake_main))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(tmp_path / "work")
    with patch.dict(os.environ, {}, clear=True):
        assert _load_project_env() is True
        assert os.environ["ANTHROPIC_API_KEY"] == "from-user-dir"


def test_working_directory_dotenv_wins_over_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doro.__main__ as cli

    fake_main = _fake_package(tmp_path, "ANTHROPIC_API_KEY=from-source-tree\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env").write_text("ANTHROPIC_API_KEY=from-cwd\n", encoding="utf-8")

    monkeypatch.setattr(cli, "__file__", str(fake_main))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    monkeypatch.chdir(work)
    with patch.dict(os.environ, {}, clear=True):
        assert _load_project_env() is True
        assert os.environ["ANTHROPIC_API_KEY"] == "from-cwd"


def test_env_file_argument_replaces_the_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "custom.env"
    explicit.write_text("ANTHROPIC_API_KEY=from-explicit-file\n", encoding="utf-8")
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    with patch.dict(os.environ, {}, clear=True):
        assert _load_project_env(env_file=str(explicit)) is True
        assert os.environ["ANTHROPIC_API_KEY"] == "from-explicit-file"


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


def _effort_args(effort: str | None = None, thinking: bool = False) -> Namespace:
    return Namespace(effort=effort, thinking=thinking)


def test_effort_cli_and_environment_precedence() -> None:
    env = {
        "OPENAI_REASONING_EFFORT": "low",
        "DORO_EFFORT": "high",
    }
    with patch.dict(os.environ, env, clear=True):
        assert _resolve_reasoning_effort(_effort_args("xhigh"), "openai") == (
            "xhigh", "--effort", "low", True
        )
        assert _resolve_reasoning_effort(_effort_args(), "openai") == (
            "low", "OPENAI_REASONING_EFFORT", "low", False
        )
    with patch.dict(os.environ, {"DORO_EFFORT": "off"}, clear=True):
        assert _resolve_reasoning_effort(_effort_args(), "anthropic") == (
            "off", "DORO_EFFORT", "off", False
        )
    with patch.dict(os.environ, {}, clear=True):
        assert _resolve_reasoning_effort(_effort_args(thinking=True), "openai") == (
            "high", "--thinking", "medium", True
        )
        assert _resolve_reasoning_effort(_effort_args(), "openai") == (
            "medium", "built-in default", "medium", False
        )


def test_effort_parser_rejects_legacy_flag_combined_with_level() -> None:
    with patch.object(sys, "argv", ["doro", "--effort", "low"]):
        assert parse_args().effort == "low"
    with (
        patch.object(sys, "argv", ["doro", "--effort", "low", "--thinking"]),
        pytest.raises(SystemExit),
    ):
        parse_args()
    with (
        patch.object(sys, "argv", ["doro", "--effort", "minimal"]),
        pytest.raises(SystemExit),
    ):
        parse_args()


@pytest.mark.asyncio
async def test_repl_effort_selector_changes_level() -> None:
    agent = Mock()
    agent.backend = "openai"
    agent.reasoning_effort = "medium"
    agent.default_reasoning_effort = "medium"
    agent._aborted = False
    agent._output_buffer = None
    agent.available_reasoning_efforts.return_value = ["auto", "medium", "high"]
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/effort", "exit"])

    with (
        patch("doro.__main__.signal.signal"),
        patch("doro.__main__.print_welcome"),
        patch("doro.__main__.print_info"),
        patch("doro.__main__.prompt_choice", new=AsyncMock(return_value="high")),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    agent.set_reasoning_effort.assert_called_once_with("high")


@pytest.mark.asyncio
async def test_repl_effort_default_resets_configuration() -> None:
    agent = Mock()
    agent._aborted = False
    agent._output_buffer = None
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/effort default", "exit"])

    with patch("doro.__main__.signal.signal"), patch("doro.__main__.print_welcome"):
        await run_repl(agent, prompt_session=prompt_session)

    agent.reset_reasoning_effort.assert_called_once_with()


@pytest.mark.asyncio
async def test_repl_model_selection_switches_by_number(monkeypatch) -> None:
    agent = Mock()
    agent.backend = "openai"
    agent.model = "alpha"
    agent._aborted = False
    agent._output_buffer = None
    agent.list_models = AsyncMock(return_value=["alpha", "beta"])
    agent.switch_model.return_value = "beta"
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["/model", "exit"])

    with (
        patch("doro.__main__.signal.signal"),
        patch("doro.__main__.print_welcome"),
        patch("doro.__main__.print_info"),
        patch(
            "doro.__main__.prompt_choice",
            new=AsyncMock(return_value="beta"),
        ),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    agent.list_models.assert_awaited_once_with()
    agent.switch_model.assert_called_once_with("beta")
    prompt_session.prompt_async.assert_awaited()
    prompt_session.prompt.assert_not_called()


@pytest.mark.parametrize("command", ["/session", "/resume"])
@pytest.mark.asyncio
async def test_repl_session_aliases_use_the_same_selector(command: str) -> None:
    agent = Mock()
    agent.backend = "openai"
    agent._aborted = False
    agent._output_buffer = None
    agent.has_conversation_history.return_value = False
    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=[command, "exit"])
    sessions = [{
        "id": "saved123",
        "model": "gpt-test",
        "messageCount": 2,
        "preview": "hello",
    }]
    choice = AsyncMock(return_value="saved123")

    with (
        patch("doro.__main__.signal.signal"),
        patch("doro.__main__.print_welcome"),
        patch("doro.__main__.print_info"),
        patch("doro.__main__._sessions_for_agent", return_value=sessions),
        patch("doro.__main__.prompt_choice", new=choice),
        patch("doro.__main__._restore_agent_session") as restore,
    ):
        await run_repl(agent, prompt_session=prompt_session)

    choice.assert_awaited_once()
    restore.assert_called_once_with(agent, "saved123")


@pytest.mark.asyncio
async def test_repl_patches_stdout_while_prompt_is_active() -> None:
    """MCP warm-up/retry output can land mid-prompt; stdout must be routed
    through prompt_toolkit so it prints above the prompt instead of shoving
    the cursor down and squashing the status toolbar."""
    import contextlib

    agent = Mock()
    agent._aborted = False
    agent._output_buffer = None
    entered: list[dict] = []

    @contextlib.contextmanager
    def fake_patch_stdout(**kwargs):
        entered.append(kwargs)
        yield

    prompt_session = Mock()
    prompt_session.prompt_async = AsyncMock(side_effect=["exit"])

    with (
        patch("doro.__main__.signal.signal"),
        patch("doro.__main__.print_welcome"),
        patch("doro.__main__.patch_stdout", fake_patch_stdout),
    ):
        await run_repl(agent, prompt_session=prompt_session)

    # raw=True: our output is pre-rendered by Rich with ANSI escapes; raw=False
    # would print them literally ("?[36m").
    assert entered == [{"raw": True}]


def test_prompt_output_guard_falls_back_without_a_console() -> None:
    """piped output / CI: prompt_toolkit can't build its output — plain
    stdout must survive instead of crashing the REPL."""
    import contextlib

    from doro.__main__ import _prompt_output_guard

    @contextlib.contextmanager
    def broken_patch_stdout():
        raise RuntimeError("no console")
        yield

    with patch("doro.__main__.patch_stdout", broken_patch_stdout):
        with _prompt_output_guard():
            print("still works")


def test_restore_helper_rejects_other_project(tmp_path: Path, monkeypatch) -> None:
    agent = Mock()
    monkeypatch.chdir(tmp_path)
    with patch("doro.__main__.load_session", return_value={
        "metadata": {
            "id": "saved123",
            "cwd": str(tmp_path / "another-project"),
        }
    }):
        with pytest.raises(ValueError, match="different project"):
            _restore_agent_session(agent, "saved123")
    agent.restore_session.assert_not_called()
