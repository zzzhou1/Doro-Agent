from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput

from mini_claude.interactive import (
    InlineSelector,
    SlashCommandCompleter,
    build_status_toolbar,
    create_history_key_bindings,
    create_repl_prompt_session,
)


def _completion_texts(completer, text: str) -> list[str]:
    return [
        completion.text
        for completion in completer.get_completions(
            Document(text=text, cursor_position=len(text)),
            CompleteEvent(completion_requested=True),
        )
    ]


def test_slash_completion_includes_commands_and_skills() -> None:
    skill = SimpleNamespace(
        name="commit",
        description="Create a commit",
        user_invocable=True,
    )
    completer = SlashCommandCompleter(lambda: [skill])

    assert _completion_texts(completer, "/mo") == ["/model"]
    assert _completion_texts(completer, "/ef") == ["/effort"]
    assert "/commit" in _completion_texts(completer, "/")
    assert _completion_texts(completer, "normal input") == []


def test_repl_prompt_session_uses_persistent_file_history(tmp_path) -> None:
    history_path = tmp_path / "input_history"
    with create_pipe_input() as pipe_input:
        session = create_repl_prompt_session(
            lambda: [],
            history_path,
            input=pipe_input,
            output=DummyOutput(),
        )

        assert isinstance(session.history, FileHistory)
        assert history_path.parent.exists()


def test_repl_prompt_session_uses_dynamic_status_toolbar(tmp_path) -> None:
    snapshot = {
        "cwd": str(tmp_path),
        "context_used": 0,
        "context_window": 128000,
        "auto_compact": True,
        "backend": "openai",
        "model": "gpt-test",
    }
    provider = Mock(return_value=snapshot)

    with create_pipe_input() as pipe_input:
        session = create_repl_prompt_session(
            lambda: [],
            tmp_path / "history",
            input=pipe_input,
            output=DummyOutput(),
            status_provider=provider,
        )
        assert callable(session.bottom_toolbar)
        fragments = session.bottom_toolbar()

    text = "".join(fragment[1] for fragment in fragments)
    assert tmp_path.name[-20:] in text
    assert "0.0%/128K auto" in text
    assert "\n" not in text
    assert "gpt-test" in text
    provider.assert_called_once_with()


def test_status_toolbar_recovers_from_provider_errors() -> None:
    def broken_provider():
        raise RuntimeError("boom")

    fragments = build_status_toolbar(broken_provider, width=40)
    text = "".join(fragment[1] for fragment in fragments)
    assert "status unavailable" in text
    assert "boom" not in text


def test_history_arrow_bindings_navigate_history() -> None:
    bindings = create_history_key_bindings()
    buffer = Mock()
    event = SimpleNamespace(current_buffer=buffer)

    bindings.get_bindings_for_keys((Keys.Up,))[0].handler(event)
    bindings.get_bindings_for_keys((Keys.Down,))[0].handler(event)

    buffer.history_backward.assert_called_once_with()
    buffer.history_forward.assert_called_once_with()


def test_slash_key_opens_completion_at_line_start() -> None:
    bindings = create_history_key_bindings()
    buffer = Mock(text="", cursor_position=0)
    event = SimpleNamespace(current_buffer=buffer)

    bindings.get_bindings_for_keys(("/",))[0].handler(event)

    buffer.insert_text.assert_called_once_with("/")
    buffer.start_completion.assert_called_once_with(select_first=False)


def test_inline_selector_arrow_keys_change_highlight() -> None:
    selector = InlineSelector(
        "Models from anthropic (2):",
        [("alpha", "1. alpha"), ("beta", "2. beta ← current")],
        initial_value="beta",
    )
    app = selector.create_application(output=DummyOutput())
    event = SimpleNamespace(app=Mock())

    assert selector.selected_value == "beta"
    app.key_bindings.get_bindings_for_keys((Keys.Down,))[0].handler(event)
    assert selector.selected_value == "alpha"
    app.key_bindings.get_bindings_for_keys((Keys.Up,))[0].handler(event)
    assert selector.selected_value == "beta"
    event.app.invalidate.assert_called()


def test_inline_selector_renders_one_highlighted_list() -> None:
    selector = InlineSelector(
        "Sessions for this project (2):",
        [("one", "1. one"), ("two", "2. two")],
    )

    rendered = selector._formatted_text()
    text = "".join(fragment[1] for fragment in rendered)
    assert "Sessions for this project (2):" in text
    assert "❯ 1. one" in text
    assert "2. two" in text
