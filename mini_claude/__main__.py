"""CLI entry point and interactive REPL."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit.patch_stdout import patch_stdout

from .agent import BACKEND_MODEL_ENV, Agent, default_model_for
from .commands import COMMAND_REGISTRY, find_command, format_help_section
from .diagnostics import build_config_report, build_doctor_report
from .interactive import create_repl_prompt_session, prompt_choice
from .memory import list_memories
from .reasoning import (
    BACKEND_REASONING_ENV,
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORTS,
    normalize_reasoning_effort,
)
from .session import get_latest_session_id, list_sessions, load_session
from .skills import discover_skills, execute_skill, get_skill_by_name, resolve_skill_prompt
from .ui import (
    live_status,
    print_error,
    print_info,
    print_panel,
    print_plan_approval_options,
    print_plan_for_approval,
    print_user_prompt,
    print_welcome,
    suspend_live_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mini-claude",
        description="Doro — a local Coding + PHM agent",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    parser.add_argument("--yolo", "-y", action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--plan", action="store_true", help="Plan mode: read-only")
    parser.add_argument("--accept-edits", action="store_true", help="Auto-approve file edits")
    parser.add_argument("--dont-ask", action="store_true", help="Auto-deny confirmations (for CI)")
    reasoning = parser.add_mutually_exclusive_group()
    reasoning.add_argument(
        "--effort",
        choices=REASONING_EFFORTS,
        default=None,
        help="Reasoning effort (default: medium)",
    )
    reasoning.add_argument("--thinking", action="store_true", help="Alias for --effort high")
    parser.add_argument("--model", "-m", default=None, help="Model to use")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--env-file", default=None, metavar="PATH", help="Load env vars from PATH instead of searching")
    parser.add_argument("--resume", action="store_true", help="Resume last session")
    parser.add_argument("--max-cost", type=float, default=None, help="Max USD spend")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agentic turns")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")
    return parser.parse_args()


def _resolve_permission_mode(args: argparse.Namespace) -> str:
    if args.yolo:
        return "bypassPermissions"
    if args.plan:
        return "plan"
    if args.accept_edits:
        return "acceptEdits"
    if args.dont_ask:
        return "dontAsk"
    return "default"


def _dotenv_candidates(env_file: str | None = None) -> list[Path]:
    """Ordered .env search path, highest priority first, de-duplicated.

    The working directory wins, then the source tree that owns this package
    (which is the project root for editable installs), then a user-level file.
    """
    if env_file:
        return [Path(env_file).expanduser()]
    paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.home() / ".mini-claude" / ".env",
    ]
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_project_env(directory: Path | None = None, env_file: str | None = None) -> bool:
    """Load .env files without overriding shell variables (first value wins).

    Passing ``directory`` restricts the search to that directory's .env.
    Otherwise ``--env-file`` or the working directory takes priority, with
    fallbacks to the package source tree and ``~/.mini-claude/.env`` so the
    installed command works from any working directory.
    """
    if directory is not None:
        candidates = [Path(directory) / ".env"]
    else:
        candidates = _dotenv_candidates(env_file)
    loaded = False
    for path in candidates:
        if path.is_file() and load_dotenv(dotenv_path=path, override=False):
            loaded = True
    return loaded


def _resolve_api_config(args: argparse.Namespace) -> tuple[str, str | None, str | None]:
    """Resolve backend, API key, and optional base URL independently."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base = os.environ.get("OPENAI_BASE_URL")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL")

    if args.api_base:
        return "openai", openai_key or anthropic_key, args.api_base
    if openai_key and openai_base:
        return "openai", openai_key, openai_base
    if anthropic_key:
        return "anthropic", anthropic_key, anthropic_base
    if openai_key:
        # The OpenAI SDK uses its official endpoint when base_url is omitted.
        return "openai", openai_key, None
    return "anthropic", None, anthropic_base


def _resolve_model(args: argparse.Namespace, backend: str) -> tuple[str, str]:
    """Resolve the model for ``backend``, plus a label saying where it came from.

    Precedence: ``--model`` > ``ANTHROPIC_MODEL`` / ``OPENAI_MODEL`` (that
    backend only) > the built-in default for that backend. The backend has to be
    resolved first for a per-backend default to exist at all: the previous code
    inferred "the user did not pick a model" by comparing the name against a
    hardcoded Claude id, which also silently rewrote an *explicit*
    ``--model claude-...`` on the OpenAI backend.

    ``MINI_CLAUDE_MODEL`` used to sit in this chain and applied to *every*
    backend, so a name meant for one provider silently travelled to the other
    (``gpt-5.6-sol`` sent to Anthropic, ``claude-opus-5`` to an OpenAI gateway).
    It was removed — set the per-backend variable instead.
    """
    backend_env_name = BACKEND_MODEL_ENV[backend]
    backend_env = os.environ.get(backend_env_name, "").strip()
    if args.model:
        return args.model, "--model"
    if backend_env:
        return backend_env, backend_env_name
    return default_model_for(backend), f"default for {backend}"


def _configured_reasoning_effort(backend: str) -> tuple[str, str]:
    """Resolve the reusable default below session and explicit CLI choices."""
    backend_env_name = BACKEND_REASONING_ENV[backend]
    backend_env = os.environ.get(backend_env_name, "").strip()
    global_env = os.environ.get("MINI_CLAUDE_EFFORT", "").strip()
    if backend_env:
        return normalize_reasoning_effort(backend_env), backend_env_name
    if global_env:
        return normalize_reasoning_effort(global_env), "MINI_CLAUDE_EFFORT"
    return DEFAULT_REASONING_EFFORT, "built-in default"


def _resolve_reasoning_effort(
    args: argparse.Namespace,
    backend: str,
) -> tuple[str, str, str, bool]:
    """Return initial effort, source, reset default, and CLI-explicit flag."""
    configured_default, default_source = _configured_reasoning_effort(backend)
    if args.effort:
        return normalize_reasoning_effort(args.effort), "--effort", configured_default, True
    if args.thinking:
        return "high", "--thinking", configured_default, True
    return configured_default, default_source, configured_default, False


@contextlib.contextmanager
def _prompt_output_guard():
    """Route stdout through prompt_toolkit while the prompt is active.

    MCP warm-up/retry messages can land mid-prompt; patch_stdout() prints them
    above the prompt and redraws the prompt + status toolbar. raw=True because
    everything we print is pre-rendered by Rich and already carries ANSI
    escapes — raw=False would write them as literal text ("?[36m"). With no
    real console attached (piped output, CI, tests) prompt_toolkit can't
    create its output — fall back to plain stdout; there is nothing on screen
    to protect.
    """
    guard = None
    try:
        guard = patch_stdout(raw=True)
        guard.__enter__()
    except Exception:
        guard = None
    try:
        yield
    finally:
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except Exception:
                pass


async def run_repl(agent: Agent, prompt_session=None) -> None:
    """Interactive REPL loop."""

    prompt_session = prompt_session or create_repl_prompt_session(
        discover_skills, status_provider=agent.get_status_snapshot
    )

    async def chat_with_status(message: str) -> None:
        def running_snapshot():
            snapshot = agent.get_status_snapshot()
            snapshot["processing"] = True
            return snapshot

        with live_status(running_snapshot):
            await agent.chat(message)

    def status_safe_input(message: str) -> str:
        with suspend_live_status():
            return input(message)

    async def confirm_fn(message: str) -> bool:
        try:
            answer = status_safe_input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False

    agent.set_confirm_fn(confirm_fn)

    async def plan_approval_fn(plan_content: str) -> dict:
        print_plan_for_approval(plan_content)
        print_plan_approval_options()
        while True:
            try:
                choice = status_safe_input("  Enter choice (1-4): ").strip()
            except EOFError:
                return {"choice": "manual-execute"}
            if choice == "1":
                return {"choice": "clear-and-execute"}
            elif choice == "2":
                return {"choice": "execute"}
            elif choice == "3":
                return {"choice": "manual-execute"}
            elif choice == "4":
                try:
                    feedback = status_safe_input("  Feedback (what to change): ").strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}
            else:
                print("  Invalid choice. Enter 1, 2, 3, or 4.")

    agent.set_plan_approval_fn(plan_approval_fn)

    validate_handlers()
    repl_ctx = _ReplContext(
        agent=agent,
        status_safe_input=status_safe_input,
    )

    sigint_count = 0

    def handle_sigint(sig, frame):
        nonlocal sigint_count
        if agent._aborted is False and agent._output_buffer is not None:
            # Agent is processing
            agent.abort()
            print("\n  (interrupted)")
            sigint_count = 0
            print_user_prompt()
        else:
            sigint_count += 1
            if sigint_count >= 2:
                print("\nBye!\n")
                sys.exit(0)
            print("\n  Press Ctrl+C again to exit.")
            print_user_prompt()

    signal.signal(signal.SIGINT, handle_sigint)
    agent.start_mcp_background()  # warm up servers while the user types
    print_welcome()

    while True:
        try:
            with _prompt_output_guard():
                line = await prompt_session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!\n")
            break

        inp = line.strip()
        sigint_count = 0

        if not inp:
            continue
        if inp in ("exit", "quit"):
            print("\nBye!\n")
            break

        # Built-in commands and skill invocations share the /-namespace
        if inp.startswith("/"):
            space_idx = inp.find(" ")
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]
            cmd_args = inp[space_idx + 1:].strip() if space_idx > 0 else ""

            # Dispatch from the unified registry (commands.COMMAND_REGISTRY).
            # Commands without takes_args fall through on trailing text so
            # "/clear please" still reaches skill lookup / chat, as before.
            spec = find_command(cmd_name)
            if spec is not None and (not cmd_args or spec.takes_args):
                await _REPL_HANDLERS[spec.name](repl_ctx, cmd_args)
                continue

            # Skill invocation: /<skill-name> [args]
            skill = get_skill_by_name(cmd_name)
            if skill and skill.user_invocable:
                print_info(f"Invoking skill: {skill.name}")
                try:
                    if skill.context == "fork":
                        result = execute_skill(skill.name, cmd_args)
                        if result:
                            await chat_with_status(f'Use the skill tool to invoke "{skill.name}" with args: {cmd_args or "(none)"}')
                    else:
                        resolved = resolve_skill_prompt(skill, cmd_args)
                        await chat_with_status(resolved)
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # Normal chat
        try:
            await chat_with_status(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))


# ─── REPL command handlers ────────────────────────────────
# One handler per CommandSpec in commands.COMMAND_REGISTRY, keyed by spec
# name. validate_handlers() (run at REPL start and in tests) keeps this map
# and the registry in lockstep, so a command cannot exist in one place only.


@dataclass
class _ReplContext:
    """What a command handler may touch: the agent and status-safe input."""

    agent: Agent
    status_safe_input: Callable[[str], str]


CommandHandler = Callable[[_ReplContext, str], Awaitable[None]]
_REPL_HANDLERS: dict[str, CommandHandler] = {}


def _handles(*names: str) -> Callable[[CommandHandler], CommandHandler]:
    def register(fn: CommandHandler) -> CommandHandler:
        for name in names:
            _REPL_HANDLERS[name] = fn
        return fn

    return register


def validate_handlers() -> None:
    """Fail loudly if the handler map and the command registry drift apart."""
    spec_names = {spec.name for spec in COMMAND_REGISTRY}
    handler_names = set(_REPL_HANDLERS)
    missing = sorted(spec_names - handler_names)
    extra = sorted(handler_names - spec_names)
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"commands without a handler: {', '.join(missing)}")
        if extra:
            problems.append(f"handlers without a CommandSpec: {', '.join(extra)}")
        raise RuntimeError("Command registry mismatch — " + "; ".join(problems))


@_handles("clear")
async def _cmd_clear(ctx: _ReplContext, _args: str) -> None:
    ctx.agent.clear_history()


@_handles("plan")
async def _cmd_plan(ctx: _ReplContext, _args: str) -> None:
    ctx.agent.toggle_plan_mode()


@_handles("cost")
async def _cmd_cost(ctx: _ReplContext, _args: str) -> None:
    ctx.agent.show_cost()


@_handles("model")
async def _cmd_model(ctx: _ReplContext, args: str) -> None:
    agent = ctx.agent
    if args:
        try:
            old_model = agent.model
            new_model = agent.switch_model(args)
            print_info(
                f"Model switched: {old_model} → {new_model} "
                f"(backend remains {agent.backend})"
            )
        except ValueError as e:
            print_error(str(e))
        return
    try:
        models = await agent.list_models()
        selector = await prompt_choice(
            f"Models from {agent.backend} ({len(models)}):",
            [
                (
                    model,
                    f"{index}. {model}"
                    + (" ← current" if model == agent.model else ""),
                )
                for index, model in enumerate(models, start=1)
            ],
            initial_value=agent.model,
        )
        if selector:
            old_model = agent.model
            new_model = agent.switch_model(
                _resolve_model_selector(selector, models)
            )
            print_info(
                f"Model switched: {old_model} → {new_model} "
                f"(backend remains {agent.backend})"
            )
    except Exception as e:
        print_error(str(e))
        print_info(
            f"Current model: {agent.model}. "
            "You can still use /model <name> to switch manually."
        )


@_handles("effort")
async def _cmd_effort(ctx: _ReplContext, args: str) -> None:
    agent = ctx.agent
    if args:
        try:
            if args.lower() == "default":
                agent.reset_reasoning_effort()
            else:
                agent.set_reasoning_effort(args)
            print_info(f"Reasoning effort: {agent.reasoning_effort}")
        except ValueError as e:
            print_error(str(e))
        return
    efforts = agent.available_reasoning_efforts()
    selector = await prompt_choice(
        f"Reasoning effort for {agent.backend}:",
        [
            (
                effort,
                effort
                + (" ← current" if effort == agent.reasoning_effort else "")
                + (
                    " (default)"
                    if effort == agent.default_reasoning_effort
                    else ""
                ),
            )
            for effort in efforts
        ],
        initial_value=agent.reasoning_effort,
    )
    if selector:
        try:
            agent.set_reasoning_effort(selector)
            print_info(f"Reasoning effort: {agent.reasoning_effort}")
        except ValueError as e:
            print_error(str(e))


@_handles("sessions")
async def _cmd_sessions(ctx: _ReplContext, _args: str) -> None:
    _print_session_list(_sessions_for_agent(ctx.agent))


@_handles("session", "resume")
async def _cmd_resume(ctx: _ReplContext, args: str) -> None:
    agent = ctx.agent
    selector: str | None = args
    sessions = _sessions_for_agent(agent)
    if not selector:
        if not sessions:
            print_info("No sessions found for this project and backend.")
            return
        selector = await prompt_choice(
            f"Sessions for this project ({len(sessions)}):",
            [
                (str(metadata["id"]), _session_choice_label(index, metadata))
                for index, metadata in enumerate(sessions, start=1)
                if metadata.get("id")
            ],
        )
        if not selector:
            return
    try:
        session_id = _resolve_session_selector(selector, sessions)
        if agent.has_conversation_history():
            try:
                answer = ctx.status_safe_input("  Replace the current conversation? (y/n): ").strip()
            except EOFError:
                answer = "n"
            if not answer.lower().startswith("y"):
                print_info("Resume cancelled.")
                return
        _restore_agent_session(agent, session_id)
    except ValueError as e:
        print_error(str(e))


@_handles("compact")
async def _cmd_compact(ctx: _ReplContext, _args: str) -> None:
    try:
        await ctx.agent.compact()
    except Exception as e:
        print_error(str(e))


@_handles("memory")
async def _cmd_memory(ctx: _ReplContext, _args: str) -> None:
    memories = list_memories()
    if not memories:
        print_info("No memories saved yet.")
    else:
        print_info(f"{len(memories)} memories:")
        for m in memories:
            print(f"    [{m.type}] {m.name} — {m.description}")


@_handles("skills")
async def _cmd_skills(ctx: _ReplContext, _args: str) -> None:
    skills = discover_skills()
    if not skills:
        print_info("No skills found. Add skills to .claude/skills/<name>/SKILL.md")
    else:
        print_info(f"{len(skills)} skills:")
        for s in skills:
            tag = f"/{s.name}" if s.user_invocable else s.name
            print(f"    {tag} ({s.source}) — {s.description}")


_MCP_STATUS_MARKERS = {
    "connected": "●",
    "connecting": "…",
    "pending": "○",
    "failed": "✗",
    "skipped": "–",
}


def _mcp_status_lines(agent: Agent) -> list[str]:
    """One row per server: state, transport, tool count, error, retry countdown."""
    statuses = agent.mcp_statuses()
    if not statuses:
        return [
            "No MCP servers configured.",
            "Add one to .mcp.json ({\"mcpServers\": {...}}), then run /mcp reconnect.",
        ]
    connected, total = agent.mcp_manager.status_counts()
    lines = [f"{connected}/{total} connected"]
    for status in statuses:
        marker = _MCP_STATUS_MARKERS.get(status["status"], "?")
        row = f"{marker} {status['name']:<16}{status['status']:<11}"
        row += f"{(status.get('transport') or '-'):<7}"
        if status["status"] == "connected":
            row += f"{status['tool_count']} tools"
        else:
            row += "-"
        if status.get("error"):
            row += f"   — {status['error']}"
        if status["status"] == "failed" and status.get("retry_in"):
            row += (
                f"   (retry in {int(status['retry_in'])}s, "
                f"attempts {status['attempts']})"
            )
        lines.append(row)
    return lines


def _mcp_tool_lines(agent: Agent) -> list[str]:
    grouped = agent.mcp_tools_by_server()
    if not grouped:
        return [
            "No MCP tools discovered.",
            "Run /mcp reconnect to (re)connect the configured servers.",
        ]
    total = sum(len(names) for names in grouped.values())
    servers = len(grouped)
    lines = [
        f"{total} tool{'' if total == 1 else 's'} across "
        f"{servers} server{'' if servers == 1 else 's'}"
    ]
    for server in sorted(grouped):
        names = grouped[server]
        lines.append(f"{server} ({len(names)})")
        lines.extend(f"  {name}" for name in names)
    return lines


@_handles("mcp")
async def _cmd_mcp(ctx: _ReplContext, args: str) -> None:
    """`/mcp`, `/mcp tools`, `/mcp reconnect`."""
    agent = ctx.agent
    action = args.strip().lower()
    if action not in ("", "tools", "reconnect"):
        print_error(
            f"Unknown /mcp subcommand {args!r}. "
            "Use /mcp, /mcp tools, or /mcp reconnect."
        )
        return
    # Read configs even if the background warm-up has not finished/failed yet,
    # so an immediately-typed /mcp still lists the configured servers.
    agent.mcp_manager.configured_servers()
    if action == "reconnect":
        print_info("Reconnecting MCP servers...")
        await agent.reconnect_mcp()
    if action == "tools":
        print_panel("MCP tools", _mcp_tool_lines(agent))
        return
    print_panel("MCP servers", _mcp_status_lines(agent))


@_handles("config")
async def _cmd_config(ctx: _ReplContext, _args: str) -> None:
    title, lines = build_config_report(ctx.agent)
    print_panel(title, lines)


@_handles("doctor")
async def _cmd_doctor(ctx: _ReplContext, _args: str) -> None:
    title, lines = build_doctor_report(ctx.agent)
    print_panel(title, lines)


def _sessions_for_agent(agent: Agent) -> list[dict]:
    return list_sessions(cwd=Path.cwd(), backend=agent.backend)


def _resolve_model_selector(selector: str, models: list[str]) -> str:
    if selector in models:
        return selector
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(models):
            return models[index - 1]
        raise ValueError(f"Model number out of range: {selector}")
    raise ValueError(
        "Model is not in the returned list. "
        "Use /model <name> to select an unlisted model directly."
    )


def _print_session_list(sessions: list[dict]) -> None:
    if not sessions:
        print_info("No sessions found for this project and backend.")
        return
    print_info(f"Sessions for this project ({len(sessions)}):")
    for index, metadata in enumerate(sessions, start=1):
        print(
            f"    {_session_choice_label(index, metadata)}"
        )


def _session_choice_label(index: int, metadata: dict) -> str:
    updated = metadata.get("updatedAt") or metadata.get("startTime") or "unknown time"
    model = metadata.get("model") or "unknown model"
    count = metadata.get("messageCount", 0)
    preview = metadata.get("preview") or "(no preview)"
    return (
        f"{index:>2}. {metadata.get('id', '?')} | {updated} | "
        f"{model} | {count} messages | {preview}"
    )


def _resolve_session_selector(selector: str, sessions: list[dict]) -> str:
    matching_ids = {str(item.get("id")) for item in sessions if item.get("id")}
    if selector in matching_ids:
        return selector
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(sessions):
            session_id = sessions[index - 1].get("id")
            if session_id:
                return str(session_id)
        raise ValueError(f"Session number out of range: {selector}")
    raise ValueError("Session not found for this project and backend")


def _restore_agent_session(agent: Agent, session_id: str) -> None:
    session = load_session(session_id)
    if session is None:
        raise ValueError(f"Session not found or unreadable: {session_id}")
    metadata = session.get("metadata") or {}
    if metadata.get("id") != session_id:
        raise ValueError("Session metadata ID does not match its file name")
    stored_cwd = metadata.get("cwd")
    if stored_cwd and os.path.normcase(str(Path(stored_cwd).resolve())) != os.path.normcase(str(Path.cwd().resolve())):
        raise ValueError("Session belongs to a different project directory")
    agent.restore_session(session)


async def _run_once(agent: Agent, prompt: str) -> None:
    """One-shot mode. Always tears down MCP server processes on the way out."""
    try:
        await agent.chat(prompt)
    finally:
        await agent.aclose()


async def _run_repl_session(agent: Agent) -> None:
    """Interactive REPL. Always tears down MCP server processes on the way out."""
    try:
        await run_repl(agent)
    finally:
        await agent.aclose()


def _force_utf8_stdio() -> None:
    """Write UTF-8 regardless of the terminal's locale.

    On Windows Python only uses UTF-8 for the *console*; as soon as stdout is a
    pipe or a file it falls back to the locale code page (cp936 on a Chinese
    system), so `mini-claude "..." > log.txt` turns every non-ASCII character
    into mojibake. The console path is already UTF-8, so reconfiguring is a
    no-op there and only fixes the redirected case.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # exotic or detached stream — nothing to reconfigure
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def main() -> None:
    _force_utf8_stdio()
    args = parse_args()
    _load_project_env(env_file=args.env_file)

    if args.help:
        print(f"""
Usage: mini-claude [options] [prompt]

Options:
  --yolo, -y          Skip all confirmation prompts (bypassPermissions mode)
  --plan              Plan mode: read-only, describe changes without executing
  --accept-edits      Auto-approve file edits, still confirm dangerous shell
  --dont-ask          Auto-deny anything needing confirmation (for CI)
  --effort LEVEL      Reasoning effort: auto/off/minimal/low/medium/high/xhigh/max
                      (default: medium; provider/model support varies)
  --thinking          Backward-compatible alias for --effort high
  --model, -m         Model to use (default per backend: anthropic=claude-opus-5,
                      openai=gpt-5.6-sol; override with ANTHROPIC_MODEL or
                      OPENAI_MODEL, which affect only their own backend)
  --api-base URL      Use OpenAI-compatible API endpoint (key via env var)
  --env-file PATH     Read env vars from PATH instead of searching for .env
  --resume            Resume the last session
  --max-cost USD      Stop when estimated cost exceeds this amount
  --max-turns N       Stop after N agentic turns
  --help, -h          Show this help

REPL commands:
{format_help_section()}

Examples:
  mini-claude "fix the bug in app.py"
  mini-claude --yolo "run all tests and fix failures"
  mini-claude --plan "how would you refactor this?"
  mini-claude --max-cost 0.50 --max-turns 20 "implement feature X"
  OPENAI_API_KEY=sk-xxx mini-claude --api-base https://aihubmix.com/v1 --model gpt-5.6-sol "hello"
  mini-claude --resume
  mini-claude  # starts interactive REPL
""")
        sys.exit(0)

    permission_mode = _resolve_permission_mode(args)
    resolved_backend, resolved_api_key, resolved_api_base = _resolve_api_config(args)
    model, model_source = _resolve_model(args, resolved_backend)
    try:
        effort, effort_source, default_effort, effort_explicit = (
            _resolve_reasoning_effort(args, resolved_backend)
        )
    except ValueError as e:
        print_error(str(e))
        sys.exit(2)

    if not resolved_api_key:
        searched = "\n".join(f"    {path}" for path in _dotenv_candidates(args.env_file))
        print_error(
            "API key is required.\n"
            "  Set ANTHROPIC_API_KEY (+ optional ANTHROPIC_BASE_URL) for Anthropic format,\n"
            "  or OPENAI_API_KEY (+ optional OPENAI_BASE_URL) for OpenAI-compatible format.\n"
            "  Also searched for a .env file in:\n"
            f"{searched}"
        )
        sys.exit(1)

    try:
        agent = Agent(
            permission_mode=permission_mode,
            model=model,
            backend=resolved_backend,
            reasoning_effort=effort,
            default_reasoning_effort=default_effort,
            reasoning_effort_explicit=effort_explicit,
            max_cost_usd=args.max_cost,
            max_turns=args.max_turns,
            api_base=resolved_api_base if resolved_backend == "openai" else None,
            anthropic_base_url=resolved_api_base if resolved_backend == "anthropic" else None,
            api_key=resolved_api_key,
            model_source=model_source,
            effort_source=effort_source,
        )
    except ValueError as e:
        print_error(str(e))
        sys.exit(2)
    print_info(
        f"Backend: {agent.backend} | model: {agent.model} ({model_source}) | "
        f"effort: {agent.reasoning_effort} ({effort_source})"
    )

    # Resume session
    if args.resume:
        session_id = get_latest_session_id(cwd=Path.cwd(), backend=agent.backend)
        if session_id:
            session = load_session(session_id)
            if session:
                try:
                    agent.restore_session(session)
                except ValueError as e:
                    print_error(str(e))
            else:
                print_info("No session found to resume.")
        else:
            print_info("No previous sessions found.")

    prompt = " ".join(args.prompt) if args.prompt else None

    if prompt:
        # One-shot mode
        try:
            asyncio.run(_run_once(agent, prompt))
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # Interactive REPL
        asyncio.run(_run_repl_session(agent))


if __name__ == "__main__":
    main()
