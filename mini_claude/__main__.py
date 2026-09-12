"""CLI entry point and interactive REPL."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .ui import print_welcome, print_user_prompt, print_error, print_info, print_plan_for_approval, print_plan_approval_options
from .session import load_session, get_latest_session_id, list_sessions
from .memory import list_memories
from .skills import discover_skills, resolve_skill_prompt, get_skill_by_name, execute_skill
from .interactive import create_repl_prompt_session, prompt_choice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mini-claude",
        description="Mini Claude Code — a minimal coding agent",
        add_help=False,
    )
    parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    parser.add_argument("--yolo", "-y", action="store_true", help="Skip all confirmation prompts")
    parser.add_argument("--plan", action="store_true", help="Plan mode: read-only")
    parser.add_argument("--accept-edits", action="store_true", help="Auto-approve file edits")
    parser.add_argument("--dont-ask", action="store_true", help="Auto-deny confirmations (for CI)")
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking")
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


async def run_repl(agent: Agent, prompt_session=None) -> None:
    """Interactive REPL loop."""

    prompt_session = prompt_session or create_repl_prompt_session(discover_skills)

    async def confirm_fn(message: str) -> bool:
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False

    agent.set_confirm_fn(confirm_fn)

    async def plan_approval_fn(plan_content: str) -> dict:
        print_plan_for_approval(plan_content)
        print_plan_approval_options()
        while True:
            try:
                choice = input("  Enter choice (1-4): ").strip()
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
                    feedback = input("  Feedback (what to change): ").strip()
                except EOFError:
                    feedback = ""
                return {"choice": "keep-planning", "feedback": feedback or None}
            else:
                print("  Invalid choice. Enter 1, 2, 3, or 4.")

    agent.set_plan_approval_fn(plan_approval_fn)

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
    print_welcome()

    while True:
        try:
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

        # REPL commands
        if inp == "/clear":
            agent.clear_history()
            continue
        if inp == "/plan":
            agent.toggle_plan_mode()
            continue
        if inp == "/cost":
            agent.show_cost()
            continue
        if inp == "/model":
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
            continue
        if inp.startswith("/model "):
            try:
                old_model = agent.model
                new_model = agent.switch_model(inp[len("/model "):])
                print_info(
                    f"Model switched: {old_model} → {new_model} "
                    f"(backend remains {agent.backend})"
                )
            except ValueError as e:
                print_error(str(e))
            continue
        if inp == "/sessions":
            _print_session_list(_sessions_for_agent(agent))
            continue
        is_resume = inp == "/resume" or inp.startswith("/resume ")
        is_session = inp == "/session" or inp.startswith("/session ")
        if is_resume or is_session:
            command = "/resume" if is_resume else "/session"
            selector = inp[len(command):].strip()
            sessions = _sessions_for_agent(agent)
            if not selector:
                if not sessions:
                    print_info("No sessions found for this project and backend.")
                    continue
                selector = await prompt_choice(
                    f"Sessions for this project ({len(sessions)}):",
                    [
                        (str(metadata["id"]), _session_choice_label(index, metadata))
                        for index, metadata in enumerate(sessions, start=1)
                        if metadata.get("id")
                    ],
                )
                if not selector:
                    continue
            try:
                session_id = _resolve_session_selector(selector, sessions)
                if agent.has_conversation_history():
                    try:
                        answer = input("  Replace the current conversation? (y/n): ").strip()
                    except EOFError:
                        answer = "n"
                    if not answer.lower().startswith("y"):
                        print_info("Resume cancelled.")
                        continue
                _restore_agent_session(agent, session_id)
            except ValueError as e:
                print_error(str(e))
            continue
        if inp == "/compact":
            try:
                await agent.compact()
            except Exception as e:
                print_error(str(e))
            continue
        if inp == "/memory":
            memories = list_memories()
            if not memories:
                print_info("No memories saved yet.")
            else:
                print_info(f"{len(memories)} memories:")
                for m in memories:
                    print(f"    [{m.type}] {m.name} — {m.description}")
            continue
        if inp == "/skills":
            skills = discover_skills()
            if not skills:
                print_info("No skills found. Add skills to .claude/skills/<name>/SKILL.md")
            else:
                print_info(f"{len(skills)} skills:")
                for s in skills:
                    tag = f"/{s.name}" if s.user_invocable else s.name
                    print(f"    {tag} ({s.source}) — {s.description}")
            continue

        # Skill invocation: /<skill-name> [args]
        if inp.startswith("/"):
            space_idx = inp.find(" ")
            cmd_name = inp[1:space_idx] if space_idx > 0 else inp[1:]
            cmd_args = inp[space_idx + 1:] if space_idx > 0 else ""
            skill = get_skill_by_name(cmd_name)
            if skill and skill.user_invocable:
                print_info(f"Invoking skill: {skill.name}")
                try:
                    if skill.context == "fork":
                        result = execute_skill(skill.name, cmd_args)
                        if result:
                            await agent.chat(f'Use the skill tool to invoke "{skill.name}" with args: {cmd_args or "(none)"}')
                    else:
                        resolved = resolve_skill_prompt(skill, cmd_args)
                        await agent.chat(resolved)
                except Exception as e:
                    if "abort" not in str(e).lower():
                        print_error(str(e))
                continue

        # Normal chat
        try:
            await agent.chat(inp)
        except Exception as e:
            if "abort" not in str(e).lower():
                print_error(str(e))


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


def main() -> None:
    args = parse_args()
    _load_project_env(env_file=args.env_file)

    if args.help:
        print("""
Usage: mini-claude [options] [prompt]

Options:
  --yolo, -y          Skip all confirmation prompts (bypassPermissions mode)
  --plan              Plan mode: read-only, describe changes without executing
  --accept-edits      Auto-approve file edits, still confirm dangerous shell
  --dont-ask          Auto-deny anything needing confirmation (for CI)
  --thinking          Enable extended thinking (Anthropic only)
  --model, -m         Model to use (default: claude-opus-4-6, or MINI_CLAUDE_MODEL env)
  --api-base URL      Use OpenAI-compatible API endpoint (key via env var)
  --env-file PATH     Read env vars from PATH instead of searching for .env
  --resume            Resume the last session
  --max-cost USD      Stop when estimated cost exceeds this amount
  --max-turns N       Stop after N agentic turns
  --help, -h          Show this help

REPL commands:
  /clear              Clear conversation history
  /plan               Toggle plan mode (read-only <-> normal)
  /cost               Show token usage and cost
  /model              List models and select one interactively
  /model NAME         Switch model within the current backend
  /session            Select and resume a saved session
  /sessions           List sessions for this project and backend
  /resume [N|ID]      Select and resume a saved session
  /compact            Manually compact conversation
  /memory             List saved memories
  /skills             List available skills
  /<skill-name>       Invoke a skill (e.g. /commit "fix types")

Examples:
  mini-claude "fix the bug in app.py"
  mini-claude --yolo "run all tests and fix failures"
  mini-claude --plan "how would you refactor this?"
  mini-claude --max-cost 0.50 --max-turns 20 "implement feature X"
  OPENAI_API_KEY=sk-xxx mini-claude --api-base https://aihubmix.com/v1 --model gpt-4o "hello"
  mini-claude --resume
  mini-claude  # starts interactive REPL
""")
        sys.exit(0)

    permission_mode = _resolve_permission_mode(args)
    model = args.model or os.environ.get("MINI_CLAUDE_MODEL", "claude-opus-4-6")
    resolved_backend, resolved_api_key, resolved_api_base = _resolve_api_config(args)
    
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
    
    # 如果用户选了 OpenAI 后端，但是模型名字还是默认的 Claude，则自动帮其修正为通用的 gpt-5.4
    if resolved_backend == "openai" and model == "claude-opus-4-6":
        model = "gpt-5.4"
    agent = Agent(
        permission_mode=permission_mode,
        model=model,
        backend=resolved_backend,
        thinking=args.thinking,
        max_cost_usd=args.max_cost,
        max_turns=args.max_turns,
        api_base=resolved_api_base if resolved_backend == "openai" else None,
        anthropic_base_url=resolved_api_base if resolved_backend == "anthropic" else None,
        api_key=resolved_api_key,
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
            asyncio.run(agent.chat(prompt))
        except Exception as e:
            print_error(str(e))
            sys.exit(1)
    else:
        # Interactive REPL
        asyncio.run(run_repl(agent))


if __name__ == "__main__":
    main()
