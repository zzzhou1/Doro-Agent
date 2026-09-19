"""Interactive terminal input, completion, selection, and history."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

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

from .commands import completion_map
from .status import format_status_toolbar

INPUT_HISTORY_FILE = Path.home() / ".doro" / "input_history"
StatusProvider = Callable[[], Mapping[str, Any]]

# Rows of live content the inline layout actually occupies: the prompt line, the
# one-row status toolbar, and one spare so a wrapped input line does not push
# the toolbar onto the cursor row.
_INLINE_LAYOUT_ROWS = 3


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

        candidates = completion_map()
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


def pin_layout_to_its_own_rows(renderer) -> None:
    """Stop prompt_toolkit from reserving every row below the cursor.

    ``Renderer.render()`` sizes the layout with::

        height = max(_min_available_height, last_height, preferred_height)

    and on Windows ``_min_available_height`` comes straight from
    ``Output.get_rows_below_cursor_position()`` — how many rows sit between the
    cursor and the bottom of the window. That is the right measure for a
    full-screen app, but Doro is an inline REPL whose UI is three rows tall, and
    the max() lets the terminal dominate:

    * Fresh terminal, cursor on row 0 of a 50-row window -> height 50. The
      prompt is drawn near the top and the status toolbar lands at the *bottom
      of the window*, with the whole screen blank between them. It only looks
      pinned once scrollback fills the window.
    * Fullscreen resize: ``_last_screen`` is dropped when the size changes, but
      ``_min_available_height`` keeps the *stale* value from the previous size,
      so the layout redisplays at the old height and leaves blank rows under
      the input line.

    Clamping it to the rows the layout actually needs makes the layout hug its
    own content at any window size, which is what an inline prompt should do.

    Re-applied before every render rather than once at startup: the renderer
    recomputes this value on each ``_request_absolute_cursor_position()`` (once
    per prompt, and again on every resize), so a single write would be
    overwritten immediately.

    Not done by printing newlines to push the cursor down: that scrolls a whole
    screen of blank lines into scrollback on every start, and the value is
    recomputed after each prompt anyway.
    """
    renderer._min_available_height = _INLINE_LAYOUT_ROWS


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


def build_status_toolbar(
    status_provider: StatusProvider, width: int | None = None
) -> list[tuple[str, str]]:
    """Return a single-row toolbar that coexists with completion menus."""
    try:
        info = format_status_toolbar(status_provider(), width)
    except Exception:
        info = "status unavailable"
    return [("class:status.info", info)]


def create_repl_prompt_session(
    skill_provider: Callable[[], Iterable[SkillLike]],
    history_path: Path | None = None,
    input=None,
    output=None,
    status_provider: StatusProvider | None = None,
) -> PromptSession:
    path = history_path or INPUT_HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # prompt_toolkit happily accepts a callable returning formatted text, but
    # its stub only spells out ``AnyFormattedText``; annotating as Any keeps the
    # call site honest without a blanket type: ignore.
    bottom_toolbar: Any = (
        (lambda: build_status_toolbar(status_provider))
        if status_provider is not None
        else None
    )
    session: PromptSession = PromptSession(
        message=[("class:prompt", "\n> ")],
        history=FileHistory(str(path)),
        completer=SlashCommandCompleter(skill_provider),
        complete_while_typing=True,
        enable_history_search=True,
        key_bindings=create_history_key_bindings(),
        bottom_toolbar=bottom_toolbar,
        # The default of 8 is dead weight here: on a short terminal the menu
        # alone could claim the whole window, and it is unrelated to what the
        # status bar needs.
        reserve_space_for_menu=_INLINE_LAYOUT_ROWS,
        style=Style.from_dict({
            "prompt": "bold ansigreen",
            "bottom-toolbar": "bg:#101010 #808080",
            "status.info": "#808080",
        }),
        input=input,
        output=output,
    )

    def _pin_layout(app) -> None:
        try:
            pin_layout_to_its_own_rows(app.renderer)
        except Exception:
            pass  # A missing renderer must not take the whole REPL down.

    # `before_render` fires inside `_redraw()`, just before `render()` reads the
    # height — the only point where a clamp actually sticks.
    # (`pre_run_callables` runs *before* the cursor-position request that
    # recomputes `_min_available_height`, so a value written there is
    # overwritten before the first frame.)
    session.app.before_render += _pin_layout
    return session


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
