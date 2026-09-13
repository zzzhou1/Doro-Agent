"""Unified slash-command registry — the single source of truth for REPL commands.

The welcome banner, ``/`` completion, ``--help`` text, and REPL dispatch all
derive from ``COMMAND_REGISTRY``. Previously each kept its own list, which is
how ``/effort`` shipped in dispatch but went missing from the welcome banner.

Adding a command:
1. Append a ``CommandSpec`` here (registry order is the welcome-banner order).
2. Register its handler in ``__main__`` with ``@_handles("name")``.
   ``validate_handlers()`` (called at REPL start and in tests) fails loudly
   if a spec has no handler or a handler has no spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    """One built-in REPL slash command.

    Attributes:
        name: Command name without the leading slash (``"effort"``).
        description: One-line summary shown as completion metadata.
        help_entries: ``(usage, description)`` pairs for the ``--help`` text;
            a command with argument forms contributes one entry per form.
        takes_args: Whether ``/name args`` is a valid invocation. Commands
            without it fall through to skill lookup / chat when given
            trailing text, preserving the pre-registry behavior.
        show_in_welcome: Whether the command appears in the welcome banner.
    """

    name: str
    description: str
    help_entries: tuple[tuple[str, str], ...]
    takes_args: bool = False
    show_in_welcome: bool = True


# Registry order defines the welcome-banner order.
COMMAND_REGISTRY: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="model",
        description="List or switch models",
        help_entries=(
            ("/model", "List models and select one interactively"),
            ("/model NAME", "Switch model within the current backend"),
        ),
        takes_args=True,
    ),
    CommandSpec(
        name="effort",
        description="Select or change reasoning effort",
        help_entries=(
            ("/effort", "Select reasoning effort interactively"),
            ("/effort LEVEL", 'Change effort; use "default" to reset configuration'),
        ),
        takes_args=True,
    ),
    CommandSpec(
        name="session",
        description="Select and resume a session",
        help_entries=(
            ("/session", "Select and resume a saved session"),
        ),
        takes_args=True,
    ),
    CommandSpec(
        name="sessions",
        description="List saved sessions",
        help_entries=(
            ("/sessions", "List sessions for this project and backend"),
        ),
    ),
    CommandSpec(
        name="resume",
        description="Select and resume a session",
        help_entries=(
            ("/resume [N|ID]", "Select and resume a saved session"),
        ),
        takes_args=True,
    ),
    CommandSpec(
        name="clear",
        description="Clear conversation history",
        help_entries=(
            ("/clear", "Clear conversation history"),
        ),
    ),
    CommandSpec(
        name="plan",
        description="Toggle plan mode",
        help_entries=(
            ("/plan", "Toggle plan mode (read-only <-> normal)"),
        ),
    ),
    CommandSpec(
        name="mcp",
        description="Show MCP server status",
        help_entries=(
            ("/mcp", "Show MCP server connection status"),
            ("/mcp tools", "List tools exposed by connected MCP servers"),
            ("/mcp reconnect", "Retry failed or pending MCP servers now"),
        ),
        takes_args=True,
    ),
    CommandSpec(
        name="cost",
        description="Show token usage and estimated cost",
        help_entries=(
            ("/cost", "Show token usage and cost"),
        ),
    ),
    CommandSpec(
        name="compact",
        description="Compact the current context",
        help_entries=(
            ("/compact", "Manually compact conversation"),
        ),
    ),
    CommandSpec(
        name="memory",
        description="List project memories",
        help_entries=(
            ("/memory", "List saved memories"),
        ),
    ),
    CommandSpec(
        name="skills",
        description="List available skills",
        help_entries=(
            ("/skills", "List available skills"),
        ),
    ),
    CommandSpec(
        name="config",
        description="Show effective configuration",
        help_entries=(
            ("/config", "Show effective configuration and where each value came from"),
        ),
    ),
    CommandSpec(
        name="doctor",
        description="Diagnose environment and API setup",
        help_entries=(
            ("/doctor", "Run environment and configuration health checks"),
        ),
    ),
)

_COMMANDS_BY_NAME = {spec.name: spec for spec in COMMAND_REGISTRY}


def find_command(name: str) -> CommandSpec | None:
    """Look up a command by name (without the leading slash)."""
    return _COMMANDS_BY_NAME.get(name)


def completion_map() -> dict[str, str]:
    """``{"/model": "List or switch models", ...}`` for the ``/`` completer."""
    return {f"/{spec.name}": spec.description for spec in COMMAND_REGISTRY}


def welcome_command_line() -> str:
    """Space-separated command names for the welcome banner."""
    return " ".join(
        f"/{spec.name}" for spec in COMMAND_REGISTRY if spec.show_in_welcome
    )


def help_entries() -> list[tuple[str, str]]:
    """Flat ``(usage, description)`` list for the ``--help`` text."""
    return [entry for spec in COMMAND_REGISTRY for entry in spec.help_entries]


# Skills are user-defined, not built-in commands (no CommandSpec, no
# handler), but they share the /-namespace, so --help documents the
# invocation form alongside the built-ins.
SKILL_HELP_ENTRY = ("/<skill-name>", 'Invoke a skill (e.g. /commit "fix types")')


def format_help_section() -> str:
    """Rendered ``REPL commands:`` body, usages aligned to the widest one."""
    entries = [*help_entries(), SKILL_HELP_ENTRY]
    width = max(len(usage) for usage, _ in entries)
    return "\n".join(f"  {usage:<{width}}  {desc}" for usage, desc in entries)
