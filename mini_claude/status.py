"""Shared terminal status-line formatting.

The interactive prompt and the streaming renderer use the same formatter so
the status bar does not jump between two different layouts when a request
starts or finishes.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from typing import Any

from prompt_toolkit.utils import get_cwidth


def terminal_width(default: int = 100) -> int:
    """Return a defensive terminal width suitable for status rendering."""
    return max(shutil.get_terminal_size((default, 24)).columns, 20)


def _display_width(text: str) -> int:
    return sum(get_cwidth(char) for char in text)


def _take_cells(text: str, cells: int, *, from_end: bool = False) -> str:
    if cells <= 0:
        return ""
    chars = reversed(text) if from_end else iter(text)
    selected: list[str] = []
    width = 0
    for char in chars:
        char_width = get_cwidth(char)
        if width + char_width > cells:
            break
        selected.append(char)
        width += char_width
    if from_end:
        selected.reverse()
    return "".join(selected)


def truncate_end(text: str, cells: int) -> str:
    """Truncate text to a display-cell budget, preserving its beginning."""
    if _display_width(text) <= cells:
        return text
    if cells <= 1:
        return "…" if cells == 1 else ""
    return _take_cells(text, cells - 1) + "…"


def truncate_start(text: str, cells: int) -> str:
    """Truncate text to a display-cell budget, preserving its latest suffix."""
    if _display_width(text) <= cells:
        return text
    if cells <= 1:
        return "…" if cells == 1 else ""
    return "…" + _take_cells(text, cells - 1, from_end=True)


def truncate_middle(text: str, cells: int) -> str:
    """Truncate a path-like string while keeping both useful ends visible."""
    if _display_width(text) <= cells:
        return text
    if cells <= 1:
        return "…" if cells == 1 else ""
    available = cells - 1
    left_cells = available // 2
    right_cells = available - left_cells
    return _take_cells(text, left_cells) + "…" + _take_cells(
        text, right_cells, from_end=True
    )


def format_token_limit(tokens: int) -> str:
    tokens = max(int(tokens), 0)
    if tokens >= 1_000_000:
        value = tokens / 1_000_000
        return f"{value:g}M"
    if tokens >= 1_000:
        value = tokens / 1_000
        return f"{value:g}K"
    return str(tokens)


def _mode_label(snapshot: Mapping[str, Any]) -> str:
    if snapshot.get("processing"):
        return "working"

    permission_mode = str(snapshot.get("permission_mode") or "default")
    mode = "plan" if permission_mode == "plan" else "normal"
    if permission_mode not in ("default", "plan"):
        mode += f" · {permission_mode}"
    effort = snapshot.get("reasoning_effort")
    if effort:
        mode += f" · think:{effort}"
    return mode


def format_status_lines(
    snapshot: Mapping[str, Any], width: int | None = None
) -> tuple[str, str, str]:
    """Build the divider, path and responsive information line."""
    width = max(width or terminal_width(), 20)

    used = max(int(snapshot.get("context_used") or 0), 0)
    limit = max(int(snapshot.get("context_window") or 0), 0)
    percent = (used / limit * 100) if limit else 0.0
    usage = f"{percent:.1f}%/{format_token_limit(limit)}"
    if snapshot.get("auto_compact", True):
        usage += " (auto)"

    left_parts = [usage]
    if width >= 72:
        left_parts.append(f"${float(snapshot.get('cost_usd') or 0):.4f}")
    if width >= 105 and snapshot.get("session_id"):
        left_parts.append(str(snapshot["session_id"]))
    left = " · ".join(left_parts)

    backend = str(snapshot.get("backend") or "unknown")
    model = str(snapshot.get("model") or "unknown")
    if width >= 55:
        right = f"({backend}) {model}"
    else:
        right = model
    if width >= 78:
        right += f" • {_mode_label(snapshot)}"

    gap = 3
    left_width = _display_width(left)
    available_right = width - left_width - gap
    if available_right < 8:
        right = ""
    else:
        right = truncate_end(right, available_right)

    if right:
        padding = width - left_width - _display_width(right)
        info = left + (" " * max(padding, 1)) + right
    else:
        info = truncate_end(left, width)

    path = truncate_middle(str(snapshot.get("cwd") or ""), width)
    return "─" * width, path, info


def format_status_toolbar(
    snapshot: Mapping[str, Any], width: int | None = None
) -> str:
    """Build a single-line status bar safe for Prompt Toolkit completion UI.

    PromptSession's bottom toolbar is fundamentally a one-row region. Feeding
    it embedded newlines makes completion-menu redraws wrap and repaint the
    whole screen on Windows terminals, so the idle layout is intentionally
    compact and Rich Live uses the same layout during work.
    """
    # Prompt Toolkit paints the bottom toolbar through the terminal's final
    # column. Windows Terminal may reserve or auto-wrap that cell, clipping the
    # final character (for example ``think:mediu``). Explicit widths are exact
    # test/layout budgets; only auto-detected terminal widths need the margin.
    requested_width = terminal_width() - 1 if width is None else width
    width = max(int(requested_width), 1)
    used = max(int(snapshot.get("context_used") or 0), 0)
    limit = max(int(snapshot.get("context_window") or 0), 0)
    percent = (used / limit * 100) if limit else 0.0
    usage = f"{percent:.1f}%/{format_token_limit(limit)}"
    if snapshot.get("auto_compact", True):
        usage += " auto"

    backend = str(snapshot.get("backend") or "unknown")
    model = str(snapshot.get("model") or "unknown")
    identity = f"{backend}:{model}" if width >= 55 else model

    right_parts = [usage]
    if width >= 115:
        right_parts.append(f"${float(snapshot.get('cost_usd') or 0):.4f}")
    if width >= 145 and snapshot.get("session_id"):
        right_parts.append(str(snapshot["session_id"]))
    right_parts.append(identity)
    if width >= 90:
        right_parts.append(_mode_label(snapshot))
    right = " · ".join(right_parts)

    if _display_width(right) > width:
        model_budget = width - _display_width(usage) - 3
        if model_budget >= 5:
            right = usage + " · " + truncate_end(model, model_budget)
        else:
            right = truncate_end(usage, width)

    cwd = str(snapshot.get("cwd") or "")
    available_path = width - _display_width(right) - 3
    if cwd and available_path >= 10:
        path = truncate_middle(cwd, available_path)
        padding = width - _display_width(path) - _display_width(right)
        return path + (" " * max(padding, 1)) + right
    return truncate_end(right, width)
