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
    ChoiceCompleter,
    SlashCommandCompleter,
    create_choice_key_bindings,
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
    assert "/commit" in _completion_texts(completer, "/")
    assert _completion_texts(completer, "normal input") == []


def test_choice_completion_filters_values_and_labels() -> None:
    completer = ChoiceCompleter([
        ("session-a", "1. session-a | fix tests"),
        ("session-b", "2. session-b | update docs"),
    ])

    assert _completion_texts(completer, "docs") == ["session-b"]
    assert _completion_texts(completer, "session-") == ["session-a", "session-b"]


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


def test_history_arrow_bindings_navigate_history() -> None:
    bindings = create_history_key_bindings()
    buffer = Mock()
    event = SimpleNamespace(current_buffer=buffer)

    bindings.get_bindings_for_keys((Keys.Up,))[0].handler(event)
    bindings.get_bindings_for_keys((Keys.Down,))[0].handler(event)

    buffer.history_backward.assert_called_once_with()
    buffer.history_forward.assert_called_once_with()


def test_choice_arrow_bindings_navigate_completion_menu() -> None:
    bindings = create_choice_key_bindings()
    buffer = Mock()
    buffer.complete_state = object()
    event = SimpleNamespace(current_buffer=buffer)

    bindings.get_bindings_for_keys((Keys.Up,))[0].handler(event)
    bindings.get_bindings_for_keys((Keys.Down,))[0].handler(event)

    buffer.complete_previous.assert_called_once_with()
    buffer.complete_next.assert_called_once_with()
