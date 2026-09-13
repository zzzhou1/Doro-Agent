"""Configuration report and health checks behind ``/config`` and ``/doctor``.

Both commands answer "what is actually in effect?", which used to require
reading ``.env``, three MCP config files, and the CLI's resolution chain by
hand. Everything shown here is non-secret: API keys are masked, never printed.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mcp_client import config_search_paths
from .session import SESSION_DIR


def mask_secret(secret: str | None) -> str:
    """Show just enough to identify a key without revealing it."""
    if not secret:
        return "(not set)"
    text = str(secret)
    if len(text) <= 8:
        return f"**** ({len(text)} chars)"
    return f"{text[:4]}…{text[-4:]} ({len(text)} chars)"


def _endpoint(agent: Any) -> str:
    if agent.backend == "openai":
        return agent.api_base or "(OpenAI default endpoint)"
    return agent.anthropic_base_url or "(Anthropic default endpoint)"


def _env_candidates() -> list[Path]:
    """Same .env search order the CLI uses (``__main__._dotenv_candidates``).

    Imported lazily: ``__main__`` imports this module for its command handlers,
    so a module-level import would be circular.
    """
    from .__main__ import _dotenv_candidates

    return _dotenv_candidates(None)


# ─── /config ────────────────────────────────────────────────


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """``1 server`` / ``2 servers`` — avoids "1 servers" in reports."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


def _server_count(path: Path) -> int:
    """Entries in one MCP config file that would become servers."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(raw, dict):
        return 0
    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        servers = raw
    return sum(
        1
        for config in servers.values()
        if isinstance(config, dict) and ("command" in config or "url" in config)
    )


def build_config_report(agent: Any) -> tuple[str, list[str]]:
    """``/config`` — every effective setting, with where it came from."""
    resolution = agent.price_resolution
    lines: list[str] = []

    lines.append(f"backend      {agent.backend}")
    lines.append(f"model        {agent.model}   [{agent.model_source}]")
    lines.append(
        f"effort       {agent.reasoning_effort}   [{agent.effort_source}]"
        f"   → thinking: {agent.thinking_mode}"
    )
    lines.append(
        f"context      {agent.context_window} tokens "
        f"(effective {agent.effective_window}, auto-compact at 85%)"
    )
    lines.append(f"permission   {agent.permission_mode}")
    lines.append(f"api base     {_endpoint(agent)}")
    lines.append(f"api key      {mask_secret(agent.api_key)}")

    price_flag = " (estimated)" if resolution.estimated else ""
    lines.append(f"pricing      {resolution.source}{price_flag}")
    lines.append(f"             {resolution.summary()}")

    lines.append("")
    lines.append("mcp configs  (later files override earlier ones)")
    for path in config_search_paths():
        if path.exists():
            lines.append(
                f"             [exists] {path}  ({_plural(_server_count(path), 'server')})"
            )
        else:
            lines.append(f"             [missing] {path}")

    configured = agent.mcp_manager.configured_servers()
    connected, total = agent.mcp_manager.status_counts()
    lines.append(
        f"mcp servers  {connected}/{total} connected, "
        f"{_plural(len(configured), 'server')} configured"
    )

    lines.append("")
    session_count = len(list(SESSION_DIR.glob("*.json"))) if SESSION_DIR.exists() else 0
    state = "exists" if SESSION_DIR.exists() else "missing"
    lines.append(f"session dir  {SESSION_DIR}  ({state}, {session_count} files)")
    for path in _env_candidates():
        if path.is_file():
            lines.append(f"env file     {path}")
            break
    else:
        lines.append("env file     (none found; using the shell environment)")

    lines.append(
        f"terminal     stdout={_encoding(sys.stdout)} "
        f"stderr={_encoding(sys.stderr)}"
    )
    lines.append(f"runtime      python {platform.python_version()} · {_prompt_toolkit_version()}")
    return "Configuration", lines


def _encoding(stream: Any) -> str:
    return getattr(stream, "encoding", None) or "unknown"


def _prompt_toolkit_version() -> str:
    try:
        import prompt_toolkit

        return f"prompt-toolkit {prompt_toolkit.__version__}"
    except Exception:
        return "prompt-toolkit (unknown)"


# ─── /doctor ────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One health check. ``status`` is ``ok`` / ``warn`` / ``fail``."""

    label: str
    status: str
    detail: str


_MARKERS = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[fail]"}


def run_checks(agent: Any) -> list[CheckResult]:
    """Everything that commonly breaks a run, in one pass, never raising."""
    checks: list[CheckResult] = []

    version = platform.python_version()
    supported = sys.version_info >= (3, 11)
    checks.append(
        CheckResult(
            "python",
            "ok" if supported else "fail",
            f"{version}" + ("" if supported else " — needs >= 3.11"),
        )
    )

    checks.append(
        CheckResult(
            "api key",
            "ok" if agent.api_key else "fail",
            f"{agent.backend} key {mask_secret(agent.api_key)}",
        )
    )

    endpoint = _endpoint(agent)
    checks.append(
        CheckResult(
            "api base",
            "ok" if endpoint.startswith(("http://", "https://")) else "warn",
            endpoint,
        )
    )

    resolution = agent.price_resolution
    checks.append(
        CheckResult(
            "pricing",
            "warn" if resolution.estimated else "ok",
            f"{agent.model}: {resolution.source}",
        )
    )

    stream_encodings = {_encoding(sys.stdout), _encoding(sys.stderr)}
    utf8 = all(str(name).lower().replace("_", "-") == "utf-8" for name in stream_encodings)
    checks.append(
        CheckResult(
            "terminal utf-8",
            "ok" if utf8 else "warn",
            ", ".join(sorted(stream_encodings))
            + ("" if utf8 else " — non-UTF-8 output can garble non-ASCII text"),
        )
    )

    session_ok = True
    session_detail = str(SESSION_DIR)
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        probe = SESSION_DIR / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as error:
        session_ok = False
        session_detail = f"{SESSION_DIR} — not writable: {error}"
    checks.append(
        CheckResult("session dir", "ok" if session_ok else "fail", session_detail)
    )

    contributing = [
        path
        for path in config_search_paths()
        if path.exists() and _server_count(path) > 0
    ]
    configured = agent.mcp_manager.configured_servers()
    if configured:
        checks.append(
            CheckResult(
                "mcp config",
                "ok",
                f"{_plural(len(configured), 'server')} from "
                f"{_plural(len(contributing), 'file')}: "
                + ", ".join(sorted(configured)),
            )
        )
    else:
        checks.append(
            CheckResult(
                "mcp config",
                "warn",
                "no MCP servers configured (checked "
                + ", ".join(str(path) for path in config_search_paths())
                + ")",
            )
        )

    statuses = agent.mcp_statuses()
    if not statuses:
        checks.append(CheckResult("mcp servers", "warn", "nothing connected yet"))
    else:
        connected, total = agent.mcp_manager.status_counts()
        broken = [s for s in statuses if s["status"] != "connected"]
        if not broken:
            checks.append(CheckResult("mcp servers", "ok", f"{connected}/{total} connected"))
        else:
            details = "; ".join(
                f"{s['name']}: {s['status']}"
                + (f" ({s['error']})" if s.get("error") else "")
                for s in broken
            )
            checks.append(
                CheckResult("mcp servers", "warn", f"{connected}/{total} connected — {details}")
            )

    try:
        import prompt_toolkit

        pt_version = prompt_toolkit.__version__
        checks.append(CheckResult("prompt-toolkit", "ok", pt_version))
    except Exception as error:
        checks.append(CheckResult("prompt-toolkit", "fail", str(error)))

    return checks


def build_doctor_report(agent: Any) -> tuple[str, list[str]]:
    """``/doctor`` — run every check and render it with a summary line."""
    checks = run_checks(agent)
    lines = [
        f"{_MARKERS[check.status]}  {check.label:<16}{check.detail}" for check in checks
    ]
    counts = {name: sum(1 for c in checks if c.status == name) for name in _MARKERS}
    lines.append("")
    summary = f"{counts['ok']} ok · {counts['warn']} warnings · {counts['fail']} failures"
    if counts["warn"] or counts["fail"]:
        summary += "  (warnings are not fatal; failures usually are)"
    lines.append(summary)
    return "Doctor", lines
