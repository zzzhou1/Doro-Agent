"""Terminal UI rendering — colored output, spinner, tool display."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.text import Text

from .commands import welcome_command_line
from .output import (
    ASSISTANT_DELTA,
    CONFIRMATION,
    DIVIDER,
    ERROR,
    INFO,
    PANEL,
    PLAN_APPROVAL,
    PLAN_OPTIONS,
    RETRY,
    SUBAGENT_END,
    SUBAGENT_START,
    TOOL_RESULT,
    TOOL_START,
    USAGE,
    USER_PROMPT,
    WARNING,
    WELCOME,
    OutputEvent,
    coordinator,
    emit,
)
from .pricing import (
    DEFAULT_CACHE_READ_DISCOUNT,
    DEFAULT_PRICING,
)
from .status import format_status_toolbar, truncate_end, truncate_start

console = Console(highlight=False)

StatusProvider = Callable[[], Mapping[str, Any]]


class _LiveStatusRenderable:
    """Dynamic Rich renderable backed by the same snapshot as the prompt bar."""

    def __init__(self, provider: StatusProvider):
        self.provider = provider

    def __rich_console__(self, _console, options):
        # Never write the terminal's final column: Windows Terminal can
        # auto-wrap it, leaving Live refreshes permanently in scrollback.
        safe_width = max(int(options.max_width) - 1, 1)
        with _live_status_lock:
            preview = truncate_start(_live_output_buffer, safe_width)
        try:
            info = format_status_toolbar(self.provider(), safe_width)
        except Exception:
            info = truncate_end("status unavailable", safe_width)
        text = Text()
        if preview:
            text.append(preview)
            text.append("\n")
        text.append(info, style="bright_black")
        yield text


_live_status: Live | None = None
_live_status_provider: StatusProvider | None = None
_live_status_lock = threading.RLock()
_live_output_buffer = ""
# True when the last terminal write left the cursor mid-line. Blocks that must
# hug the line above them (the turn divider, and the usage footer under it) ask
# for a terminating newline only when one is actually needed; the usual
# ``"" if _flush_live_output() else "\n"`` idiom always inserts a blank line,
# which is too loose for a footer.
_line_open = False


def _block_prefix() -> str:
    """Newline-terminate an unfinished line, but never insert a blank one."""
    global _line_open
    if _flush_live_output():
        _line_open = False
        return ""
    if _line_open:
        _line_open = False
        return "\n"
    return ""


def live_status_active() -> bool:
    with _live_status_lock:
        return _live_status is not None


def start_live_status(provider: StatusProvider) -> bool:
    """Pin status below streaming output; return whether Live was started."""
    global _live_output_buffer, _live_status, _live_status_provider
    with _live_status_lock:
        if _live_status is not None or not console.is_terminal:
            return False
        _live_output_buffer = ""
        live = Live(
            _LiveStatusRenderable(provider),
            console=console,
            refresh_per_second=4,
            transient=True,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        _live_status = live
        _live_status_provider = provider
        try:
            live.start(refresh=True)
            return True
        except Exception:
            _live_status = None
            _live_status_provider = None
            try:
                live.stop()
            except Exception:
                pass
            return False


def refresh_live_status() -> None:
    with _live_status_lock:
        if _live_status is not None:
            _live_status.refresh()


def stop_live_status() -> None:
    """Remove the live bar, preserving any unfinished streamed line."""
    global _live_output_buffer, _live_status, _live_status_provider, _line_open
    with _live_status_lock:
        live = _live_status
        pending = _live_output_buffer
        _live_output_buffer = ""
        _live_status = None
        _live_status_provider = None
    if live is not None:
        try:
            live.stop()
        except Exception:
            pass
        if pending:
            console.print(
                pending, end="", markup=False, highlight=False, soft_wrap=True
            )
            _line_open = not pending.endswith("\n")


def _flush_live_output() -> bool:
    """Commit the pending assistant line before another UI block is printed."""
    global _live_output_buffer, _line_open
    with _live_status_lock:
        live = _live_status
        pending = _live_output_buffer
        _live_output_buffer = ""
    if live is None or not pending:
        return False
    console.print(
        pending + "\n", end="", markup=False, highlight=False, soft_wrap=True
    )
    _line_open = False
    try:
        live.refresh()
    except Exception:
        pass
    return True


@contextmanager
def live_status(provider: StatusProvider) -> Iterator[None]:
    started = start_live_status(provider)
    try:
        yield
    finally:
        if started:
            stop_live_status()


@contextmanager
def suspend_live_status() -> Iterator[None]:
    """Temporarily release the terminal for blocking confirmation prompts."""
    with _live_status_lock:
        provider = _live_status_provider
    if provider is None:
        yield
        return
    stop_live_status()
    try:
        yield
    finally:
        start_live_status(provider)
# ─── Basic output ──────────────────────────────────────────


def print_welcome() -> None:
    emit(OutputEvent(WELCOME))


def _render_welcome(_event: OutputEvent) -> None:
    console.print(
        "\n  [bold cyan]Doro[/bold cyan][dim] — A local Coding + PHM agent[/dim]\n"
    )
    console.print("[dim]  Type your request, or 'exit' to quit.[/dim]")
    console.print(f"[dim]  Commands: {welcome_command_line()}[/dim]\n")


def print_user_prompt() -> None:
    emit(OutputEvent(USER_PROMPT))


def _render_user_prompt(_event: OutputEvent) -> None:
    console.print("\n[bold green]> [/bold green]", end="")


def print_assistant_text(text: str) -> None:
    emit(OutputEvent(ASSISTANT_DELTA, text=text))


def _render_assistant_delta(event: OutputEvent) -> None:
    global _live_output_buffer, _line_open
    text = event.text
    with _live_status_lock:
        live = _live_status
        if live is not None:
            combined = _live_output_buffer + text
            boundary = combined.rfind("\n")
            if boundary >= 0:
                completed = combined[: boundary + 1]
                _live_output_buffer = combined[boundary + 1 :]
            else:
                completed = ""
                _live_output_buffer = combined
    if live is None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if text:
            _line_open = not text.endswith("\n")
        return
    # Only complete lines go through Console.print. Printing every partial
    # token with end="" makes Rich append its Live renderable to that token.
    if completed:
        console.print(
            completed, end="", markup=False, highlight=False, soft_wrap=True
        )
    # Anything left in the buffer is still mid-line, either pending in the live
    # bar or about to be committed by _flush_live_output.
    _line_open = bool(_live_output_buffer)
    try:
        live.refresh()
    except Exception:
        pass


def print_tool_call(name: str, inp: dict) -> None:
    emit(OutputEvent(TOOL_START, data={"name": name, "input": inp}))


def _render_tool_start(event: OutputEvent) -> None:
    name = str(event.data.get("name", ""))
    inp = event.data.get("input") or {}
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, dict(inp))
    prefix = "" if _flush_live_output() else "\n"
    console.print(
        f"{prefix}  [yellow]{icon} {escape(name)}[/yellow][dim] {escape(summary)}[/dim]"
    )


def print_tool_result(name: str, result: str) -> None:
    emit(OutputEvent(TOOL_RESULT, data={"name": name, "result": result}))


def _render_tool_result(event: OutputEvent) -> None:
    name = str(event.data.get("name", ""))
    result = str(event.data.get("result", ""))
    _flush_live_output()
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    lines = "\n".join("  " + line for line in truncated.split("\n"))
    # markup=False: tool output is model/server-controlled text; brackets in
    # it must render literally instead of being eaten as style tags.
    console.print(lines, style="dim", markup=False, highlight=False)


def _print_file_change_result(_name: str, result: str) -> None:
    # Diff content is tool output: style per line, never markup (brackets in
    # the diff would otherwise be parsed as style tags and disappear).
    lines = result.split("\n")
    console.print("  " + lines[0], style="dim", markup=False, highlight=False)

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        if line.startswith("@@"):
            console.print("  " + line, style="cyan", markup=False, highlight=False)
        elif line.startswith("- "):
            console.print("  " + line, style="red", markup=False, highlight=False)
        elif line.startswith("+ "):
            console.print("  " + line, style="green", markup=False, highlight=False)
        else:
            console.print("  " + line, style="dim", markup=False, highlight=False)
    if len(content_lines) > max_display:
        console.print(f"  ... ({len(content_lines) - max_display} more lines)", style="dim")


def print_error(msg: str) -> None:
    emit(OutputEvent(ERROR, text=msg))


def _render_error(event: OutputEvent) -> None:
    prefix = "" if _flush_live_output() else "\n"
    console.print(
        f"{prefix}  Error: {event.text}", style="red", markup=False, highlight=False
    )


def print_confirmation(command: str) -> None:
    emit(OutputEvent(CONFIRMATION, text=command))


def _render_confirmation(event: OutputEvent) -> None:
    prefix = "" if _flush_live_output() else "\n"
    console.print(
        f"{prefix}  [yellow]⚠ Dangerous command:[/yellow] [white]{escape(event.text)}[/white]"
    )


def print_warning(msg: str) -> None:
    emit(OutputEvent(WARNING, text=msg))


def _render_warning(event: OutputEvent) -> None:
    prefix = "" if _flush_live_output() else "\n"
    console.print(
        f"{prefix}  ⚠ {event.text}", style="yellow", markup=False, highlight=False
    )


def print_divider() -> None:
    emit(OutputEvent(DIVIDER))


def _render_divider(_event: OutputEvent) -> None:
    # Hugs whatever came before: no blank line, just terminate an unfinished
    # one. The usage footer is emitted immediately after and must sit directly
    # under the rule, so this adds nothing below it either.
    console.print(f"{_block_prefix()}[dim]  {'─' * 50}[/dim]")


# Legacy rate constants, kept so ``estimate_cost_usd`` stays callable without a
# model in hand (tests and tools rely on it). Per-model pricing lives in
# ``pricing.py``; the agent now prices through ``pricing.compute_cost`` and only
# falls back to the discount-based primitive when no rate card is available.
PRICE_INPUT_PER_M = DEFAULT_PRICING.input_per_m
PRICE_CACHE_WRITE_PER_M = DEFAULT_PRICING.cache_write_per_m
PRICE_OUTPUT_PER_M = DEFAULT_PRICING.output_per_m


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_discount: float = DEFAULT_CACHE_READ_DISCOUNT,
) -> float:
    """Estimated spend.

    ``input_tokens`` is the *total* input — the cached portion is included and
    then discounted here, because callers track one input number (that is what
    the context-window accounting needs) rather than two.
    """
    fresh = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    return (
        fresh * PRICE_INPUT_PER_M
        + cache_write_tokens * PRICE_CACHE_WRITE_PER_M
        + cache_read_tokens * PRICE_INPUT_PER_M * cache_read_discount
        + output_tokens * PRICE_OUTPUT_PER_M
    ) / 1_000_000


def format_token_usage(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> str:
    """``Tokens: 8837 in (8814 cached, 23 written) / 2 out``.

    The cached marker matters: a cached prompt reports a near-zero fresh input
    count, which looks like a bug unless the cached tokens are shown next to it.

    Read and write are reported **separately and never summed into one figure**.
    They point in opposite directions — a cache read is discounted (0.1x-0.5x the
    input rate), a cache write is surcharged (1.25x) — so adding them together
    presents a premium as a saving. It also fabricates a hit rate: a backend that
    books every newly-seen token as a cache write while reporting ``input_tokens``
    of 0 (the rightapi.ai gateway does exactly this) turns the merged label into a
    literal "100% cached" on every single turn, new content included.
    """
    parts = []
    if cache_read_tokens:
        parts.append(f"{cache_read_tokens} cached")
    if cache_write_tokens:
        parts.append(f"{cache_write_tokens} written")
    suffix = f" ({', '.join(parts)})" if parts else ""
    return f"Tokens: {input_tokens} in{suffix} / {output_tokens} out"


def format_usage_line(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost: float | None = None,
    estimated: bool = False,
    *,
    calls: int = 0,
) -> str:
    """One usage row: token breakdown plus an optional cost figure.

    Reasoning tokens are only reported by OpenAI-compatible endpoints; Anthropic
    folds them into ``output_tokens``, so the marker is omitted when the count is
    zero rather than printing a misleading literal 0.

    ``calls`` is the billable API-call count behind the row. It only makes sense
    for the *round* row: a turn that used a tool bills twice, so the round sum
    can exceed the context window, and printing the call count is what makes
    that legible instead of looking like double-counting.
    """
    line = format_token_usage(
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
    )
    if reasoning_tokens:
        line += f" · {reasoning_tokens} reasoning"
    if calls:
        line += f" · {calls} call{'s' if calls != 1 else ''}"
    if cost is not None:
        line += f" (~${cost:.4f}{' estimated' if estimated else ''})"
    return line


def format_cached_percentage(cache_read_tokens: int, input_tokens: int) -> str:
    """``97% cached`` — cache reads as a share of the prompt, else ``""``.

    Only cache *reads* count as cached here. Cache writes do not: they are
    surcharged, and on a gateway that books every newly-seen token as a write
    while reporting ``input_tokens`` of 0 (rightapi.ai does this) counting them
    would print a literal 100% on content the model had never seen. Leaving
    them out keeps the figure honest — the same turn reads 97%, not 100%.
    """
    if input_tokens <= 0 or cache_read_tokens <= 0:
        return ""
    share = min(cache_read_tokens, input_tokens) / input_tokens * 100
    return f"{share:.0f}% cached"


def format_round_footer(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost: float | None = None,
    calls: int = 0,
) -> str:
    """The one-line end-of-turn footer: this round's spend, nothing else.

    ``Tokens: 21124 in (97% cached) / 113 out · 2 calls (~$0.0103)``

    Session totals and the ``estimated`` qualifier are deliberately absent —
    they belong in ``/cost``, which is asked for rather than printed at every
    turn boundary. The read/write split is collapsed to a percentage here for
    width; ``/cost`` still shows both figures apart.
    """
    cached = format_cached_percentage(cache_read_tokens, input_tokens)
    suffix = f" ({cached})" if cached else ""
    line = f"Tokens: {input_tokens} in{suffix} / {output_tokens} out"
    if reasoning_tokens:
        line += f" · {reasoning_tokens} reasoning"
    if calls:
        line += f" · {calls} call{'s' if calls != 1 else ''}"
    if cost is not None:
        line += f" (~${cost:.4f})"
    return line


def print_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_discount: float = DEFAULT_CACHE_READ_DISCOUNT,
    *,
    reasoning_tokens: int = 0,
    session_cost: float | None = None,
    round_usage: Mapping[str, Any] | None = None,
    model: str = "",
    price_source: str = "",
    estimated: bool = False,
) -> None:
    """End-of-turn usage line.

    The positional arguments stay the session totals so existing callers keep
    working; ``round_usage`` adds the per-turn slice, and ``session_cost`` /
    ``estimated`` carry a pricing-table result. When those are omitted the
    renderer falls back to the legacy discount-based estimate.
    """
    data: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_discount": cache_read_discount,
        "reasoning_tokens": reasoning_tokens,
        "model": model,
        "price_source": price_source,
        "estimated": estimated,
    }
    if session_cost is not None:
        data["session_cost"] = session_cost
    if round_usage is not None:
        data["round"] = dict(round_usage)
    emit(OutputEvent(USAGE, data=data))


def _render_usage(event: OutputEvent) -> None:
    data = event.data

    round_usage = data.get("round")
    if isinstance(round_usage, Mapping):
        # End-of-turn footer: one line, this round's spend, sitting directly
        # under the divider. The trailing newline adds a little air before the
        # next prompt without loosening the divider/footer pair above it.
        line = format_round_footer(
            int(round_usage.get("input", 0)),
            int(round_usage.get("output", 0)),
            int(round_usage.get("cache_read", 0)),
            int(round_usage.get("reasoning", 0)),
            float(round_usage.get("cost") or 0.0),
            calls=int(round_usage.get("calls", 0)),
        )
        console.print(
            f"{_block_prefix()}  {line}\n",
            style="dim",
            markup=False,
            highlight=False,
        )
        return

    # Legacy shape (no round slice): the session total stands alone.
    input_tokens = int(data.get("input_tokens", 0))
    output_tokens = int(data.get("output_tokens", 0))
    cache_read_tokens = int(data.get("cache_read_tokens", 0))
    cache_write_tokens = int(data.get("cache_write_tokens", 0))
    reasoning_tokens = int(data.get("reasoning_tokens", 0))
    estimated = bool(data.get("estimated", False))

    session_cost = data.get("session_cost")
    if session_cost is None:
        session_cost = estimate_cost_usd(
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            float(data.get("cache_read_discount", DEFAULT_CACHE_READ_DISCOUNT)),
        )
    session_cost = float(session_cost)

    prefix = "" if _flush_live_output() else "\n"
    console.print(
        prefix
        + "  "
        + format_usage_line(
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            session_cost,
            estimated,
        ),
        style="dim",
        markup=False,
        highlight=False,
    )


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    emit(OutputEvent(RETRY, data={
        "attempt": attempt,
        "max_retries": max_retries,
        "reason": reason,
    }))


def _render_retry(event: OutputEvent) -> None:
    prefix = "" if _flush_live_output() else "\n"
    attempt = event.data.get("attempt", 0)
    max_retries = event.data.get("max_retries", 0)
    reason = escape(str(event.data.get("reason", "")))
    console.print(f"{prefix}  [yellow]↻ Retry {attempt}/{max_retries}: {reason}[/yellow]")


def print_info(msg: str) -> None:
    emit(OutputEvent(INFO, text=msg))


def _render_info(event: OutputEvent) -> None:
    prefix = "" if _flush_live_output() else "\n"
    console.print(
        f"{prefix}  ℹ {event.text}", style="cyan", markup=False, highlight=False
    )


# ─── Titled reports (/mcp, /config, /doctor) ────────────────


def print_panel(title: str, lines: Iterable[str]) -> None:
    """Titled multi-line report routed through the coordinator.

    Command output is emitted as an event rather than printed directly so it
    cannot land in the middle of a live status refresh.
    """
    emit(OutputEvent(PANEL, data={"title": title, "lines": list(lines)}))


def _render_panel(event: OutputEvent) -> None:
    title = str(event.data.get("title", ""))
    lines = event.data.get("lines") or []
    prefix = "" if _flush_live_output() else "\n"
    console.print(
        f"{prefix}\n  [bold cyan]{escape(title)}[/bold cyan]", highlight=False
    )
    # Report bodies are pre-formatted plain text (paths, model ids, errors) —
    # markup=False so any brackets in them survive instead of being eaten.
    for line in lines:
        console.print(
            "  " + str(line), markup=False, highlight=False
        )


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "Thinking") -> None:
    if live_status_active():
        refresh_live_status()
        return
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        frame = 0
        with coordinator.synchronized():
            sys.stdout.write(f"\n  {SPINNER_FRAMES[0]} {label}...")
            sys.stdout.flush()
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            frame = (frame + 1) % len(SPINNER_FRAMES)
            with coordinator.synchronized():
                sys.stdout.write(f"\r  {SPINNER_FRAMES[frame]} {label}...")
                sys.stdout.flush()

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    if live_status_active():
        refresh_live_status()
        return
    global _spinner_thread
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    with coordinator.synchronized():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


# ─── Plan approval display ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    emit(OutputEvent(PLAN_APPROVAL, text=plan_content))


def _render_plan_approval(event: OutputEvent) -> None:
    _flush_live_output()
    console.print("\n  [cyan]━━━ Plan for Approval ━━━[/cyan]")
    lines = event.text.split("\n")
    max_lines = 60
    for line in lines[:max_lines]:
        console.print("  " + line, style="white", markup=False, highlight=False)
    if len(lines) > max_lines:
        console.print(f"  ... ({len(lines) - max_lines} more lines)", style="dim")
    console.print("  [cyan]━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n")


def print_plan_approval_options() -> None:
    emit(OutputEvent(PLAN_OPTIONS))


def _render_plan_options(_event: OutputEvent) -> None:
    _flush_live_output()
    console.print("  [yellow]Choose an option:[/yellow]")
    console.print("    [white]1) Yes, clear context and execute[/white][dim] — fresh start with auto-accept edits[/dim]")
    console.print("    [white]2) Yes, and execute[/white][dim] — keep context, auto-accept edits[/dim]")
    console.print("    [white]3) Yes, manually approve edits[/white][dim] — keep context, confirm each edit[/dim]")
    console.print("    [white]4) No, keep planning[/white][dim] — provide feedback to revise[/dim]")


# ─── Sub-agent display ──────────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    emit(OutputEvent(SUBAGENT_START, data={
        "agent_type": agent_type,
        "description": description,
    }))


def _render_subagent_start(event: OutputEvent) -> None:
    prefix = "" if _flush_live_output() else "\n"
    agent_type = escape(str(event.data.get("agent_type", "")))
    description = escape(str(event.data.get("description", "")))
    console.print(
        f"{prefix}  [magenta]┌─ Sub-agent [{agent_type}]: {description}[/magenta]"
    )


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    emit(OutputEvent(SUBAGENT_END, data={"agent_type": agent_type}))


def _render_subagent_end(event: OutputEvent) -> None:
    _flush_live_output()
    agent_type = escape(str(event.data.get("agent_type", "")))
    console.print(f"  [magenta]└─ Sub-agent [{agent_type}] completed[/magenta]")


# ─── Event dispatch ───────────────────────────────────────
# The single renderer behind output.coordinator: every event emitted
# anywhere in the app funnels through here, serialized by the coordinator's
# lock, so no output path can bypass the live-status buffer.

_RENDERERS = {
    ASSISTANT_DELTA: _render_assistant_delta,
    TOOL_START: _render_tool_start,
    TOOL_RESULT: _render_tool_result,
    WARNING: _render_warning,
    INFO: _render_info,
    ERROR: _render_error,
    CONFIRMATION: _render_confirmation,
    RETRY: _render_retry,
    USAGE: _render_usage,
    PANEL: _render_panel,
    DIVIDER: _render_divider,
    WELCOME: _render_welcome,
    USER_PROMPT: _render_user_prompt,
    PLAN_APPROVAL: _render_plan_approval,
    PLAN_OPTIONS: _render_plan_options,
    SUBAGENT_START: _render_subagent_start,
    SUBAGENT_END: _render_subagent_end,
}


def render_event(event: OutputEvent) -> None:
    renderer = _RENDERERS.get(event.kind)
    if renderer is not None:
        renderer(event)


coordinator.set_renderer(render_event)


# ─── Tool icons and summaries ───────────────────────────────

_TOOL_ICONS = {
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "🔧",
    "list_files": "📁",
    "grep_search": "🔍",
    "run_shell": "💻",
    "skill": "⚡",
    "agent": "🤖",
}


def _get_tool_icon(name: str) -> str:
    return _TOOL_ICONS.get(name, "🔨")


def _get_tool_summary(name: str, inp: dict) -> str:
    if name == "read_file":
        return inp.get("file_path", "")
    if name == "write_file":
        return inp.get("file_path", "")
    if name == "edit_file":
        return inp.get("file_path", "")
    if name == "list_files":
        return inp.get("pattern", "")
    if name == "grep_search":
        return f'"{inp.get("pattern", "")}" in {inp.get("path", ".")}'
    if name == "run_shell":
        cmd = inp.get("command", "")
        return cmd[:60] + "..." if len(cmd) > 60 else cmd
    if name == "skill":
        return inp.get("skill_name", "")
    if name == "agent":
        return f'[{inp.get("type", "general")}] {inp.get("description", "")}'
    return ""
