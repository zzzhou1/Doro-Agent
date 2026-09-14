"""Unified terminal output dispatch — one coordinator, one renderer, one lock.

Every piece of terminal output is an :class:`OutputEvent` emitted to the
module-level :data:`coordinator`, which serializes events through a single
lock and hands them to one renderer (registered by ``ui.py`` at import
time). Modules outside ``ui`` never write to stdout directly — they emit
semantic events, so streamed assistant text, tool logs, warnings, and usage
lines cannot interleave with each other or bypass the live-status buffer.

Event kinds are plain string constants so any module can emit without
importing an enum. The canonical kinds:

- ``assistant_delta`` — streamed assistant text chunk (``text``)
- ``tool_start`` / ``tool_result`` — tool call display (``data``: name/input/result)
- ``warning`` / ``info`` / ``error`` — diagnostic lines (``text``)
- ``confirmation`` / ``retry`` — agent control-flow lines
- ``usage`` — token/cost line (``data``: token counts + discount)
- ``panel`` — titled multi-line report (``data``: title + lines); used by
  ``/mcp``, ``/config`` and ``/doctor`` so command output stays on the
  coordinator path instead of being printed straight to the terminal
- ``divider`` — turn separator
- ``welcome`` / ``user_prompt`` — REPL chrome
- ``plan_approval`` / ``plan_options`` — plan-mode display
- ``subagent_start`` / ``subagent_end`` — sub-agent display
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# ─── Event kinds ──────────────────────────────────────────

ASSISTANT_DELTA = "assistant_delta"
TOOL_START = "tool_start"
TOOL_RESULT = "tool_result"
WARNING = "warning"
INFO = "info"
ERROR = "error"
CONFIRMATION = "confirmation"
RETRY = "retry"
USAGE = "usage"
PANEL = "panel"
DIVIDER = "divider"
WELCOME = "welcome"
USER_PROMPT = "user_prompt"
PLAN_APPROVAL = "plan_approval"
PLAN_OPTIONS = "plan_options"
SUBAGENT_START = "subagent_start"
SUBAGENT_END = "subagent_end"


@dataclass(frozen=True)
class OutputEvent:
    """One unit of terminal output. ``data`` carries structured payloads."""

    kind: str
    text: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


Renderer = Callable[[OutputEvent], None]


def _fallback_renderer(event: OutputEvent) -> None:
    """Plain-stdout renderer used before ui registers, or if it fails.

    Output is diagnostic — it must never take the agent loop down with it.
    """
    text = event.text
    if not text and event.data:
        text = " ".join(str(v) for v in event.data.values())
    if text:
        try:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception:
            pass


class OutputCoordinator:
    """Serializes every output event through one lock to one renderer."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._renderer: Renderer = _fallback_renderer

    def set_renderer(self, renderer: Renderer) -> None:
        with self._lock:
            self._renderer = renderer

    def emit(self, event: OutputEvent) -> None:
        with self._lock:
            try:
                self._renderer(event)
            except Exception:
                _fallback_renderer(event)

    @contextmanager
    def synchronized(self) -> Iterator[None]:
        """Hold the output lock across several raw writes (spinner frames)."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


coordinator = OutputCoordinator()


def emit(event: OutputEvent) -> None:
    coordinator.emit(event)


def emit_info(text: str) -> None:
    emit(OutputEvent(INFO, text=text))


def emit_warning(text: str) -> None:
    emit(OutputEvent(WARNING, text=text))
