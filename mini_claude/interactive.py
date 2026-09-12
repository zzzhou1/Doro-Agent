"""Interactive terminal input, completion, selection, and history."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Protocol

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_completions
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
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


def create_history_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("/")
    def _start_command_completion(event) -> None:
        buffer = event.current_buffer
        at_line_start = buffer.cursor_position == 0 and not buffer.text
        buffer.insert_text("/")
        if at_line_start:
            buffer.start_completion(select_first=False)

    @bindings.add("up", filter=~has_completions)
    def _history_previous(event) -> None:
        event.current_buffer.history_backward()

    @bindings.add("down", filter=~has_completions)
    def _history_next(event) -> None:
        event.current_buffer.history_forward()

    return bindings


class InlineSelector:
    """An in-place arrow-key selector that renders the choices as one list."""

    def __init__(
        self,
        title: str,
        options: list[tuple[str, str]],
        initial_value: str | None = None,
    ):
        self.title = title
        self.options = options
        self.selected_index = 0
        if initial_value is not None:
            for index, (value, _) in enumerate(options):
                if value == initial_value:
                    self.selected_index = index
                    break

    @property
    def selected_value(self) -> str:
        return self.options[self.selected_index][0]

    def _formatted_text(self):
        fragments = [
            ("class:selector.title", f"\n  {self.title}\n"),
            ("class:selector.hint", "  ↑/↓ move · Enter select · Esc cancel\n\n"),
        ]
        for index, (_, label) in enumerate(self.options):
            if index == self.selected_index:
                fragments.append(("class:selector.selected", f"  ❯ {label}\n"))
            else:
                fragments.append(("class:selector.normal", f"    {label}\n"))
        return fragments

    def create_application(self, input=None, output=None) -> Application:
        bindings = KeyBindings()

        @bindings.add("up")
        def _previous(event) -> None:
            self.selected_index = (self.selected_index - 1) % len(self.options)
            event.app.invalidate()

        @bindings.add("down")
        def _next(event) -> None:
            self.selected_index = (self.selected_index + 1) % len(self.options)
            event.app.invalidate()

        @bindings.add("enter")
        def _select(event) -> None:
            event.app.exit(result=self.selected_value)

        @bindings.add("escape")
        @bindings.add("c-c")
        def _cancel(event) -> None:
            event.app.exit(result=None)

        control = FormattedTextControl(
            text=self._formatted_text,
            focusable=True,
            show_cursor=False,
        )
        window = Window(
            content=control,
            dont_extend_height=True,
            wrap_lines=True,
            always_hide_cursor=True,
        )
        return Application(
            layout=Layout(window, focused_element=window),
            key_bindings=bindings,
            style=Style.from_dict({
                "selector.title": "bold ansicyan",
                "selector.hint": "ansibrightblack",
                "selector.selected": "reverse bold",
                "selector.normal": "",
            }),
            full_screen=False,
            erase_when_done=False,
            input=input,
            output=output,
        )


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
    title: str,
    options: list[tuple[str, str]],
    initial_value: str | None = None,
    input=None,
    output=None,
) -> str | None:
    """Render one in-place list and return the arrow-key-selected value."""
    if not options:
        return None
    try:
        selector = InlineSelector(
            title=title,
            options=options,
            initial_value=initial_value,
        )
        result = await selector.create_application(input=input, output=output).run_async()
    except (EOFError, KeyboardInterrupt):
        return None
    return result
