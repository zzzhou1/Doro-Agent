"""Interactive terminal input, completion, selection, and history."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_completions
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


INPUT_HISTORY_FILE = Path.home() / ".mini-claude" / "input_history"

BUILTIN_COMMANDS = {
    "/clear": "Clear conversation history",
    "/compact": "Compact the current context",
    "/cost": "Show token usage and estimated cost",
    "/memory": "List project memories",
    "/model": "List or switch models",
    "/plan": "Toggle plan mode",
    "/resume": "Select and resume a session",
    "/session": "Select and resume a session",
    "/sessions": "List saved sessions",
    "/skills": "List available skills",
}


class SkillLike(Protocol):
    name: str
    description: str
    user_invocable: bool


class SlashCommandCompleter(Completer):
    """Complete built-in slash commands and user-invocable skills."""

    def __init__(self, skill_provider: Callable[[], Iterable[SkillLike]]):
        self._skill_provider = skill_provider

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or any(char.isspace() for char in text):
            return

        candidates = dict(BUILTIN_COMMANDS)
        try:
            for skill in self._skill_provider():
                if skill.user_invocable:
                    candidates.setdefault(f"/{skill.name}", skill.description)
        except Exception:
            pass

        for command, description in sorted(candidates.items()):
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display_meta=description,
                )


class ChoiceCompleter(Completer):
    """Complete a value while displaying a richer label."""

    def __init__(self, options: list[tuple[str, str]]):
        self._options = options

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        normalized = text.casefold()
        for value, label in self._options:
            if not normalized or normalized in value.casefold() or normalized in label.casefold():
                yield Completion(value, start_position=-len(text), display=label)


def create_history_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("up", filter=~has_completions)
    def _history_previous(event) -> None:
        event.current_buffer.history_backward()

    @bindings.add("down", filter=~has_completions)
    def _history_next(event) -> None:
        event.current_buffer.history_forward()

    return bindings


def create_choice_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("up")
    def _previous_choice(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state is None:
            buffer.start_completion(select_last=True)
        else:
            buffer.complete_previous()

    @bindings.add("down")
    def _next_choice(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state is None:
            buffer.start_completion(select_first=True)
        else:
            buffer.complete_next()

    @bindings.add("enter")
    def _accept_selection(event) -> None:
        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.current_completion is not None:
            buffer.apply_completion(state.current_completion)
        buffer.validate_and_handle()

    return bindings


def create_repl_prompt_session(
    skill_provider: Callable[[], Iterable[SkillLike]],
    history_path: Path | None = None,
    input=None,
    output=None,
) -> PromptSession:
    path = history_path or INPUT_HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        message=[("class:prompt", "\n> ")],
        history=FileHistory(str(path)),
        completer=SlashCommandCompleter(skill_provider),
        complete_while_typing=True,
        enable_history_search=True,
        key_bindings=create_history_key_bindings(),
        style=Style.from_dict({"prompt": "bold ansigreen"}),
        input=input,
        output=output,
    )


async def prompt_choice(
    message: str,
    options: list[tuple[str, str]],
    input=None,
    output=None,
) -> str | None:
    """Select with arrow keys or type a value; Enter accepts the highlighted item."""
    if not options:
        return None

    def _show_choices() -> None:
        get_app().current_buffer.start_completion(select_first=True)

    try:
        session = PromptSession(
            message=message,
            completer=ChoiceCompleter(options),
            complete_while_typing=True,
            key_bindings=create_choice_key_bindings(),
            input=input,
            output=output,
        )
        result = await session.prompt_async(pre_run=_show_choices)
    except (EOFError, KeyboardInterrupt):
        return None
    return result.strip() or None
