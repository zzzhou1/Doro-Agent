"""Terminal UI rendering — colored output, spinner, tool display."""

from __future__ import annotations

import sys
import threading
import time

from rich.console import Console

console = Console(highlight=False)

# ─── Basic output ──────────────────────────────────────────


def print_welcome() -> None:
    console.print("\n  [bold cyan]Mini Claude[/bold cyan][dim] — A minimal coding agent[/dim]\n")
    console.print("[dim]  Type your request, or 'exit' to quit.[/dim]")
    console.print("[dim]  Commands: /model /session /sessions /resume /clear /plan /cost /compact /memory /skills[/dim]\n")


def print_user_prompt() -> None:
    console.print("\n[bold green]> [/bold green]", end="")


def print_assistant_text(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def print_tool_call(name: str, inp: dict) -> None:
    icon = _get_tool_icon(name)
    summary = _get_tool_summary(name, inp)
    console.print(f"\n  [yellow]{icon} {name}[/yellow][dim] {summary}[/dim]")


def print_tool_result(name: str, result: str) -> None:
    if (name in ("edit_file", "write_file")) and not result.startswith("Error"):
        _print_file_change_result(name, result)
        return
    max_len = 500
    truncated = result
    if len(result) > max_len:
        truncated = result[:max_len] + f"\n  ... ({len(result)} chars total)"
    lines = "\n".join("  " + l for l in truncated.split("\n"))
    console.print(f"[dim]{lines}[/dim]")


def _print_file_change_result(_name: str, result: str) -> None:
    lines = result.split("\n")
    console.print(f"[dim]  {lines[0]}[/dim]")

    max_display = 40
    content_lines = lines[1:]
    display_lines = content_lines[:max_display]

    for line in display_lines:
        if not line.strip():
            continue
        if line.startswith("@@"):
            console.print(f"[cyan]  {line}[/cyan]")
        elif line.startswith("- "):
            console.print(f"[red]  {line}[/red]")
        elif line.startswith("+ "):
            console.print(f"[green]  {line}[/green]")
        else:
            console.print(f"[dim]  {line}[/dim]")
    if len(content_lines) > max_display:
        console.print(f"[dim]  ... ({len(content_lines) - max_display} more lines)[/dim]")


def print_error(msg: str) -> None:
    console.print(f"\n  [red]Error: {msg}[/red]")


def print_confirmation(command: str) -> None:
    console.print(f"\n  [yellow]⚠ Dangerous command:[/yellow] [white]{command}[/white]")


def print_divider() -> None:
    console.print(f"\n[dim]  {'─' * 50}[/dim]")


# USD per million tokens. Anthropic bills prompt-cache *writes* at 1.25x the
# input rate, which is why the cached portion is priced apart from fresh input
# instead of being lumped in at the full rate.
PRICE_INPUT_PER_M = 3.0
PRICE_CACHE_WRITE_PER_M = 3.75
PRICE_OUTPUT_PER_M = 15.0

# Cache *reads* are discounted by provider, and the multipliers differ enough to
# matter: Anthropic reads cached input at 0.1x the input rate, OpenAI-compatible
# endpoints at 0.5x. Pricing an OpenAI cache read at Anthropic's 0.1x would
# understate real spend by 5x, so the multiplier travels with the backend rather
# than being baked into a single constant.
CACHE_READ_DISCOUNT = {"anthropic": 0.1, "openai": 0.5}
DEFAULT_CACHE_READ_DISCOUNT = 0.1


def cache_read_discount_for(backend: str) -> float:
    """Cache-read discount multiplier for a backend id (``openai``/``anthropic``)."""
    return CACHE_READ_DISCOUNT.get(backend, DEFAULT_CACHE_READ_DISCOUNT)


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
    """``Tokens: 8814 in (8814 cached) / 2 out``.

    The cached marker matters: a cached prompt reports a near-zero fresh input
    count, which looks like a bug unless the cached tokens are shown next to it.
    """
    cached = cache_read_tokens + cache_write_tokens
    suffix = f" ({cached} cached)" if cached else ""
    return f"Tokens: {input_tokens} in{suffix} / {output_tokens} out"


def print_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_discount: float = DEFAULT_CACHE_READ_DISCOUNT,
) -> None:
    total = estimate_cost_usd(
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        cache_read_discount,
    )
    usage = format_token_usage(
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
    )
    console.print(f"\n[dim]  {usage} (~${total:.4f})[/dim]")


def print_retry(attempt: int, max_retries: int, reason: str) -> None:
    console.print(f"\n  [yellow]↻ Retry {attempt}/{max_retries}: {reason}[/yellow]")


def print_info(msg: str) -> None:
    console.print(f"\n  [cyan]ℹ {msg}[/cyan]")


# ─── Spinner ──────────────────────────────────────────────

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_spinner_thread: threading.Thread | None = None
_spinner_stop = threading.Event()


def start_spinner(label: str = "Thinking") -> None:
    global _spinner_thread
    if _spinner_thread is not None:
        return
    _spinner_stop.clear()

    def _run() -> None:
        frame = 0
        sys.stdout.write(f"\n  {SPINNER_FRAMES[0]} {label}...")
        sys.stdout.flush()
        while not _spinner_stop.is_set():
            time.sleep(0.08)
            frame = (frame + 1) % len(SPINNER_FRAMES)
            sys.stdout.write(f"\r  {SPINNER_FRAMES[frame]} {label}...")
            sys.stdout.flush()

    _spinner_thread = threading.Thread(target=_run, daemon=True)
    _spinner_thread.start()


def stop_spinner() -> None:
    global _spinner_thread
    if _spinner_thread is None:
        return
    _spinner_stop.set()
    _spinner_thread.join(timeout=1)
    _spinner_thread = None
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ─── Plan approval display ──────────────────────────────────


def print_plan_for_approval(plan_content: str) -> None:
    console.print("\n  [cyan]━━━ Plan for Approval ━━━[/cyan]")
    lines = plan_content.split("\n")
    max_lines = 60
    for line in lines[:max_lines]:
        console.print(f"  [white]{line}[/white]")
    if len(lines) > max_lines:
        console.print(f"[dim]  ... ({len(lines) - max_lines} more lines)[/dim]")
    console.print("  [cyan]━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n")


def print_plan_approval_options() -> None:
    console.print("  [yellow]Choose an option:[/yellow]")
    console.print("    [white]1) Yes, clear context and execute[/white][dim] — fresh start with auto-accept edits[/dim]")
    console.print("    [white]2) Yes, and execute[/white][dim] — keep context, auto-accept edits[/dim]")
    console.print("    [white]3) Yes, manually approve edits[/white][dim] — keep context, confirm each edit[/dim]")
    console.print("    [white]4) No, keep planning[/white][dim] — provide feedback to revise[/dim]")


# ─── Sub-agent display ──────────────────────────────────────


def print_sub_agent_start(agent_type: str, description: str) -> None:
    console.print(f"\n  [magenta]┌─ Sub-agent [{agent_type}]: {description}[/magenta]")


def print_sub_agent_end(agent_type: str, _description: str) -> None:
    console.print(f"  [magenta]└─ Sub-agent [{agent_type}] completed[/magenta]")


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
