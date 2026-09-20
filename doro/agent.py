"""Agent core loop — dual backend (Anthropic + OpenAI compatible), streaming,
4-layer compression, plan mode, sub-agents, budget control.
Mirrors Claude Code's agent architecture."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import anthropic
import openai

from .mcp_client import McpManager
from .memory import (
    MemoryPrefetch,
    format_memories_for_injection,
    start_memory_prefetch,
)
from .output import emit_warning
from .pricing import PriceResolution, cache_read_discount_for, compute_cost, resolve_pricing
from .prompt import build_system_prompt
from .reasoning import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORTS,
    normalize_reasoning_effort,
    reasoning_fallback_for,
)
from .session import save_session
from .subagent import get_sub_agent_config
from .tools import (
    CONCURRENCY_SAFE_TOOLS,
    ToolDef,
    check_permission,
    execute_tool,
    get_active_tool_definitions,
    tool_definitions,
)
from .ui import (
    format_usage_line,
    print_assistant_text,
    print_confirmation,
    print_cost,
    print_divider,
    print_info,
    print_panel,
    print_retry,
    print_sub_agent_end,
    print_sub_agent_start,
    print_tool_call,
    print_tool_result,
    start_spinner,
    stop_spinner,
)

# ─── Retry with exponential backoff ──────────────────────────


def _is_retryable(error: Exception) -> bool:
    #判断当前发生的异常是否值得重试。
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (429, 503, 529):
        return True
    msg = str(error)
    if "overloaded" in msg or "ECONNRESET" in msg or "ETIMEDOUT" in msg:
        return True
    return False
#如果 HTTP 状态码是 429（请求过于频繁）、503（服务不可用）或 529（过载），返回 True。
# 如果错误信息中包含 "overloaded"（服务过载）、"ECONNRESET"（连接被重置）或 "ETIMEDOUT"（连接超时），也返回 True。
# 其他硬性错误（如 400 参数错误、401 鉴权失败）则直接放弃重试，避免浪费资源。

async def _with_retry(fn, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = min(1000 * (2 ** attempt), 30000) / 1000 + (hash(str(time.time())) % 1000) / 1000
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            reason = f"HTTP {status}" if status else (getattr(error, "code", None) or "network error")
            print_retry(attempt + 1, max_retries, reason)
            await asyncio.sleep(delay)


def _anthropic_usage_totals(usage) -> tuple[int, int, int]:
    """Split an Anthropic ``usage`` payload into (total_input, cache_write, cache_read).

    ``input_tokens`` is NOT the whole input. It counts only the portion that
    missed the prompt cache; when the system prompt and tool definitions are
    served from cache — which is the normal case after the first turn — it can
    legitimately read 0 while thousands of tokens were read from cache:

        input_tokens=0  cache_creation_input_tokens=23  cache_read_input_tokens=8814

    Everything the model actually read has to be summed back together, because
    this number also drives context-window accounting. Treating it as 0 would
    disable auto-compaction and let the conversation grow past the window.
    """
    fresh = getattr(usage, "input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return fresh + cache_write + cache_read, cache_write, cache_read


def _openai_usage_totals(usage) -> tuple[int, int, int]:
    """Split an OpenAI ``usage`` payload into (total_input, output, cache_read).

    The mirror image of ``_anthropic_usage_totals``, and the difference is the
    easy thing to get wrong: OpenAI's ``prompt_tokens`` is already the *whole*
    prompt. The cached prefix is a **subset** of it, reported separately in
    ``prompt_tokens_details.cached_tokens``:

        prompt_tokens=2372  prompt_tokens_details.cached_tokens=1984

    So the cached count is returned for *pricing* only. Adding it back into the
    input total the way the Anthropic branch must would double-count it and
    inflate ``last_input_token_count`` — the number every context-management
    threshold is keyed off.

    Cache writes are deliberately not read: OpenAI does not surcharge them, and
    the fields some gateways expose for it (``cache_write_tokens``,
    ``cached_creation_tokens``) are already inside ``prompt_tokens``, so they
    are billed as fresh input — conservative, never a silent undercount.
    """
    total_input = getattr(usage, "prompt_tokens", 0) or 0
    output = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    # A gateway may report a cached count larger than the prompt it belongs to;
    # clamp so the discounted portion can never exceed the total it discounts.
    cache_read = min(cache_read, total_input)
    return total_input, output, cache_read


def _openai_reasoning_tokens(usage) -> int:
    """Reasoning tokens reported by OpenAI-compatible reasoning models.

    These are already inside ``completion_tokens``; the count is surfaced for
    visibility only and must never be added to the totals. Anthropic has no
    equivalent field — thinking tokens land in ``output_tokens`` — so this
    returns 0 there and callers must not treat 0 as "reasoning was off".
    """
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return min(int(reasoning), int(completion))


# ─── Default model per backend ──────────────────────────────

# One shared default cannot work: a Claude model name means nothing to an
# OpenAI-compatible gateway and a GPT name means nothing to Anthropic, so a
# single hardcoded default always breaks one of the two backends.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5.6-sol",
}

# Per-backend override env var, consulted by ``resolve_default_model``.
# Deliberately per-backend only: the global ``DORO_MODEL`` override was
# removed, because a model name written while one provider was active leaked
# silently into the other backend.
BACKEND_MODEL_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
}


def default_model_for(backend: str) -> str:
    """Built-in default model for a backend (unknown backends fall back to Anthropic)."""
    return DEFAULT_MODELS.get(backend, DEFAULT_MODELS["anthropic"])


def resolve_default_model(backend: str, override: str | None = None) -> str:
    """Pick a model for ``backend``: explicit > per-backend env var > built-in default."""
    if override and override.strip():
        return override.strip()
    env_name = BACKEND_MODEL_ENV.get(backend)
    env_value = os.environ.get(env_name, "") if env_name else ""
    if env_value.strip():
        return env_value.strip()
    return default_model_for(backend)


# ─── Model context windows ──────────────────────────────────

# One window for every model. The per-model table that used to live here was
# guessing at gateway aliases (`gpt-5.6-sol`, `claude-opus-5`) whose real limits
# are not public, and a wrong-but-plausible number is worse than a deliberate
# default: it silently moves the point where auto-compaction fires, and a table
# lookup by exact name meant every dated or repackaged alias missed it anyway.
DEFAULT_CONTEXT_WINDOW = 272_000


def _resolve_context_window_override() -> int | None:
    """``DORO_CONTEXT_WINDOW`` as a positive int, else ``None``.

    A malformed value is ignored rather than fatal: a typo in `.env` must not
    stop the CLI from starting, and the default is always usable.
    """
    raw = os.environ.get("DORO_CONTEXT_WINDOW", "").strip()
    if not raw:
        return None
    try:
        value = int(raw.replace("_", "").replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def _get_context_window(model: str) -> int:
    """Context window for ``model`` — one shared default, `.env`-overridable.

    ``model`` is accepted and ignored so every call site keeps its current
    shape, and so a per-model table could be reintroduced without touching them.
    """
    return _resolve_context_window_override() or DEFAULT_CONTEXT_WINDOW


# ─── Thinking support detection ─────────────────────────────


def _get_max_output_tokens(model: str) -> int:
    m = model.lower()
    if any(x in m for x in ("opus-5", "opus-4-6")):
        return 64000
    if "sonnet-4-6" in m:
        return 32000
    if any(x in m for x in ("opus-4", "sonnet-4", "haiku-4")):
        return 32000
    return 16384


# ─── Convert tools to OpenAI format ─────────────────────────


def _to_openai_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# ─── Multi-tier compression constants ────────────────────────

SNIPPABLE_TOOLS = {"read_file", "grep_search", "list_files", "run_shell"}
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIP_THRESHOLD = 0.60
MICROCOMPACT_IDLE_S = 5 * 60  # 5 minutes
KEEP_RECENT_RESULTS = 3


# ─── Agent ───────────────────────────────────────────────────


class Agent:
    def __init__(
        self,
        *,
        permission_mode: str = "default",
        model: str | None = None,
        backend: str | None = None,
        api_base: str | None = None,
        anthropic_base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        default_reasoning_effort: str | None = None,
        reasoning_effort_explicit: bool = False,
        max_cost_usd: float | None = None,
        max_turns: int | None = None,
        confirm_fn: Callable[[str], Awaitable[bool]] | None = None,
        custom_system_prompt: str | None = None,
        custom_tools: list[ToolDef] | None = None,
        is_sub_agent: bool = False,
        model_source: str = "unknown",
        effort_source: str = "unknown",
    ):
        if backend not in (None, "anthropic", "openai"):
            raise ValueError("backend must be 'anthropic' or 'openai'")

        self.permission_mode = permission_mode
        self.backend = backend or ("openai" if api_base else "anthropic")
        self.use_openai = self.backend == "openai"
        # The backend is resolved first so an omitted model can pick up that
        # backend's own default instead of one shared hardcoded name.
        self.model = resolve_default_model(self.backend, model)
        # Where the model / effort came from, for /config and /doctor. These are
        # resolved by the CLI (which sees --model and the env vars); a bare
        # Agent() has no way to know, hence the "unknown" default.
        self.model_source = model_source
        self.default_reasoning_effort = normalize_reasoning_effort(
            default_reasoning_effort or DEFAULT_REASONING_EFFORT
        )
        selected_effort = reasoning_effort or self.default_reasoning_effort
        if thinking:
            selected_effort = "high"
        self.reasoning_effort = normalize_reasoning_effort(selected_effort)
        self.effort_source = effort_source
        self._reasoning_effort_explicit = reasoning_effort_explicit or bool(thinking)
        self.thinking = self.reasoning_effort not in ("auto", "off")
        self.api_base = api_base
        self.anthropic_base_url = anthropic_base_url
        self.api_key = api_key
        self.is_sub_agent = is_sub_agent
        self.tools = custom_tools or tool_definitions
        self.max_cost_usd = max_cost_usd
        self.max_turns = max_turns
        self.confirm_fn = confirm_fn
        self.context_window = _get_context_window(self.model)
        self.effective_window = self.context_window - 20000
        self.session_id = uuid.uuid4().hex[:8]
        self.session_start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._last_user_preview = ""

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_reasoning_tokens = 0
        self.last_input_token_count = 0
        # Count of billable API calls made by this agent instance (and by any
        # sub-agent whose spend was folded back in). Kept because the round
        # figures are sums over calls: a turn that used a tool bills twice, and
        # without the count "20760 in" next to a 10k context reads like a bug.
        self._api_calls = 0
        # Cache reads are discounted differently per backend (0.1x Anthropic vs
        # 0.5x OpenAI), so the multiplier is resolved once here. It is retained
        # for the legacy discount path; real cost now goes through the per-model
        # rate card resolved below.
        self._cache_read_discount = cache_read_discount_for(self.backend)
        self.refresh_pricing()
        # "This round" figures are derived as (current totals - this snapshot),
        # taken at the top of chat(). Deriving instead of incrementing means
        # compaction and sub-agent spend are included automatically without
        # touching every accumulation site.
        self._round_base = self._usage_totals()
        self._round_api_base = 0
        self.current_turns = 0
        self.last_api_call_time = 0.0

        # Abort support
        self._aborted = False
        self._current_task: asyncio.Task | None = None

        # Permission whitelist
        self._confirmed_paths: set[str] = set()

        # Plan mode state
        self._pre_plan_mode: str | None = None
        self._plan_file_path: str | None = None
        self._plan_approval_fn: Callable[[str], Awaitable[dict]] | None = None
        self._context_cleared: bool = False  # Set when plan approval clears context

        # Endpoint compatibility is checked by the real API. Private gateways
        # and aliases make local model-name capability tables unreliable.
        self._thinking_mode = self._resolve_thinking_mode()

        # Output buffer (sub-agents capture output)
        self._output_buffer: list[str] | None = None

        # Read-before-edit: track file read timestamps (absolutePath → mtime)
        self._read_file_state: dict[str, float] = {}

        # MCP integration
        self._mcp_manager = McpManager()
        self._mcp_initialized = False

        # Memory recall state — semantic prefetch per user turn
        self._already_surfaced_memories: set[str] = set()
        self._session_memory_bytes = 0

        # Separate message histories
        self._anthropic_messages: list[dict] = []
        self._openai_messages: list[dict] = []

        # Build system prompt
        self._base_system_prompt = custom_system_prompt or build_system_prompt()
        if self.permission_mode == "plan":
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
        else:
            self._system_prompt = self._base_system_prompt

        # Initialize clients. Exactly one of them is live — which one is fixed
        # by the backend, so both stay Optional and each method that uses one
        # asserts its own invariant.
        self._openai_client: openai.AsyncOpenAI | None = None
        self._anthropic_client: anthropic.AsyncAnthropic | None = None
        if self.use_openai:
            self._openai_client = openai.AsyncOpenAI(base_url=api_base, api_key=api_key)
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        else:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if anthropic_base_url:
                kwargs["base_url"] = anthropic_base_url
            self._anthropic_client = anthropic.AsyncAnthropic(**kwargs)

    def _resolve_thinking_mode(self) -> str:
        if self.reasoning_effort == "auto":
            return "auto"
        if self.reasoning_effort == "off":
            return "disabled"
        if self.use_openai:
            return "reasoning"
        return "adaptive"

    def available_reasoning_efforts(self) -> list[str]:
        return list(REASONING_EFFORTS)

    def set_reasoning_effort(self, effort: str, *, explicit: bool = True) -> str:
        normalized = normalize_reasoning_effort(effort)
        self.reasoning_effort = normalized
        self._reasoning_effort_explicit = explicit
        self.thinking = normalized not in ("auto", "off")
        self._thinking_mode = self._resolve_thinking_mode()
        return normalized

    def reset_reasoning_effort(self) -> str:
        return self.set_reasoning_effort(
            self.default_reasoning_effort,
            explicit=False,
        )

    def _openai_reasoning_params(self) -> dict[str, str]:
        if self.reasoning_effort == "auto":
            return {}
        wire_effort = "none" if self.reasoning_effort == "off" else self.reasoning_effort
        return {"reasoning_effort": wire_effort}

    def _anthropic_reasoning_params(self) -> dict[str, Any]:
        if self.reasoning_effort == "auto":
            return {}
        if self.reasoning_effort == "off":
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.reasoning_effort},
        }

    async def _with_reasoning_fallback(
        self,
        call: Callable[[dict[str, bool]], Awaitable[Any]],
    ) -> Any:
        """Run an API call with visible, compatibility-only effort fallback."""
        fallback_count = 0
        while True:
            stream_state = {"started": False}
            try:
                return await _with_retry(lambda state=stream_state: call(state))
            except Exception as error:
                next_effort = reasoning_fallback_for(error, self.reasoning_effort)
                if stream_state["started"] or next_effort is None or fallback_count >= 2:
                    raise
                # Try one lower controllable level, then omit the field. This
                # keeps a restrictive gateway from causing a long retry walk.
                if fallback_count == 1 and next_effort != "auto":
                    next_effort = "auto"
                previous = self.reasoning_effort
                explicit = self._reasoning_effort_explicit
                self.set_reasoning_effort(next_effort, explicit=explicit)
                self.effort_source = f"automatic fallback from {previous}"
                stop_spinner()
                source = "Requested" if explicit else "Configured"
                emit_warning(
                    f"{source} reasoning effort '{previous}' is unsupported by the endpoint; "
                    f"using '{next_effort}'."
                )
                fallback_count += 1

    @property
    def is_processing(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def _build_side_query(self):
        """Build a sideQuery callable for memory recall, works with both backends."""
        if self._anthropic_client:
            client = self._anthropic_client
            model = self.model
            async def _sq(system: str, user_message: str) -> str:
                resp = await client.messages.create(
                    model=model, max_tokens=256, system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            return _sq
        if self._openai_client:
            client = self._openai_client
            model = self.model
            async def _sq_oai(system: str, user_message: str) -> str:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                )
                return resp.choices[0].message.content or "" if resp.choices else ""
            return _sq_oai
        return None

    def abort(self) -> None:
        self._aborted = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def set_confirm_fn(self, fn: Callable[[str], Awaitable[bool]]) -> None:
        self.confirm_fn = fn

    def set_plan_approval_fn(self, fn: Callable[[str], Awaitable[dict]]) -> None:
        self._plan_approval_fn = fn

    # ─── Plan mode toggle ────────────────────────────────────

    def toggle_plan_mode(self) -> str:
        if self.permission_mode == "plan":
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(f"Exited plan mode → {self.permission_mode} mode")
            return self.permission_mode
        else:
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info(f"Entered plan mode. Plan file: {self._plan_file_path}")
            return "plan"

    def _usage_totals(self) -> dict[str, int]:
        """Session-cumulative counters, reasoning tokens included."""
        return {
            "input": self.total_input_tokens,
            "output": self.total_output_tokens,
            "cache_read": self.total_cache_read_tokens,
            "cache_write": self.total_cache_write_tokens,
            "reasoning": self.total_reasoning_tokens,
        }

    def get_token_usage(self) -> dict:
        return dict(self._usage_totals())

    def _round_tokens(self) -> dict[str, int]:
        """Spend since the current ``chat()`` started (this turn's slice)."""
        base = self._round_base
        return {
            key: value - base.get(key, 0)
            for key, value in self._usage_totals().items()
        }

    def _round_calls(self) -> int:
        """Billable API calls made since the current ``chat()`` started.

        A single user turn with a tool call bills more than once (one call to
        decide the tool, another to answer with the result), and compaction or a
        sub-agent adds calls of its own. Surfacing the count next to the round
        token sum is what keeps ``20760 in`` from reading as a bug when the
        context window only holds ~10k.
        """
        return self._api_calls - self._round_api_base

    def get_api_call_count(self) -> int:
        """Session-cumulative billable API calls (for diagnostics)."""
        return self._api_calls

    def refresh_pricing(self) -> PriceResolution:
        """Re-resolve the rate card (after a model switch); returns the result."""
        self._price_resolution = resolve_pricing(self.model, backend=self.backend)
        return self._price_resolution

    @property
    def price_resolution(self) -> PriceResolution:
        return self._price_resolution

    def _cost_of(self, tokens: dict[str, int]) -> float:
        return compute_cost(
            self._price_resolution.pricing,
            tokens.get("input", 0),
            tokens.get("output", 0),
            tokens.get("cache_read", 0),
            tokens.get("cache_write", 0),
        )

    def get_status_snapshot(self) -> dict[str, Any]:
        """Return non-sensitive state for the interactive terminal status bar."""
        mcp_connected, mcp_total = self._mcp_manager.status_counts()
        return {
            "cwd": str(Path.cwd()),
            "context_used": self.last_input_token_count,
            "context_window": self.context_window,
            "auto_compact": True,
            "cost_usd": self._get_current_cost_usd(),
            "round_cost_usd": self._cost_of(self._round_tokens()),
            "cost_estimated": self._price_resolution.estimated,
            "session_id": self.session_id,
            "backend": self.backend,
            "model": self.model,
            "permission_mode": self.permission_mode,
            "thinking_mode": self._thinking_mode,
            "reasoning_effort": self.reasoning_effort,
            "mcp_connected": mcp_connected,
            "mcp_total": mcp_total,
            "processing": self.is_processing,
        }

    async def aclose(self) -> None:
        """Release external resources: MCP server subprocesses and HTTP clients."""
        try:
            await self._mcp_manager.disconnect_all()
        except Exception:
            pass

    # ─── Main entry point ────────────────────────────────────

    def start_mcp_background(self) -> None:
        """Begin MCP server connections in the background (REPL startup).

        Connections then warm up while the user types; the first chat turn
        awaits whatever is left via wait_ready().
        """
        if not self.is_sub_agent:
            self._mcp_manager.start_background()

    @property
    def mcp_manager(self) -> McpManager:
        return self._mcp_manager

    @property
    def thinking_mode(self) -> str:
        """``auto`` / ``disabled`` / ``adaptive`` / ``enabled`` / ``reasoning``."""
        return self._thinking_mode

    def mcp_statuses(self) -> list[dict]:
        """Per-server MCP connection state (for ``/mcp`` and ``/doctor``)."""
        return self._mcp_manager.server_statuses()

    def mcp_tools_by_server(self) -> dict[str, list[str]]:
        """Raw MCP tool names grouped by server (for ``/mcp tools``)."""
        return self._mcp_manager.tools_by_server()

    async def reconnect_mcp(self) -> bool:
        """``/mcp reconnect`` — retry failed/pending servers immediately.

        Also re-runs tool injection so a server that came up after the first
        turn becomes callable without restarting the REPL.
        """
        if self.is_sub_agent:
            return False
        connected = await self._mcp_manager.load_and_connect(force=True)
        definitions = self._mcp_manager.get_tool_definitions()
        if definitions:
            known = {tool.get("name") for tool in self.tools}
            self.tools.extend(
                tool for tool in definitions if tool.get("name") not in known
            )
        self._mcp_initialized = connected
        return connected

    async def _ensure_mcp_initialized(self) -> None:
        """Discover MCP tools, retaining retryability after transient failures."""
        if self._mcp_initialized or self.is_sub_agent:
            return
        try:
            fully_connected = await self._mcp_manager.wait_ready()
            mcp_defs = self._mcp_manager.get_tool_definitions()
            if mcp_defs:
                known_names = {tool.get("name") for tool in self.tools}
                self.tools.extend(
                    tool for tool in mcp_defs if tool.get("name") not in known_names
                )
            self._mcp_initialized = fully_connected
        except Exception as e:
            self._mcp_initialized = False
            emit_warning(f"[mcp] Init failed: {e}")

    async def chat(self, user_message: str) -> None:
        self._last_user_preview = " ".join(user_message.split())[:120]
        # Snapshot the session counters so /cost and the end-of-turn line can
        # show "this round" without incrementing anything extra at each call.
        self._round_base = self._usage_totals()
        self._round_api_base = self._api_calls
        await self._ensure_mcp_initialized()

        self._aborted = False
        
        # 步骤 B：根据后端路由，将请求分发给具体的执行循环
        coro = self._chat_openai(user_message) if self.use_openai else self._chat_anthropic(user_message)
        self._current_task = asyncio.current_task()
        try:
            await coro
        except asyncio.CancelledError:
            self._aborted = True # 支持用户按下 Ctrl+C 强行终止
        finally:
            self._current_task = None
            
        # 步骤 C：自动保存会话
        if not self.is_sub_agent:
            # Divider first, usage footer second: the footer reads as a closing
            # note for the answer above, so it belongs under the rule. Emitting
            # it here (rather than at the loop's exit) also covers the paths
            # that leave the loop by budget break or Ctrl+C.
            print_divider()
            self._emit_turn_cost()
            self._auto_save()

    # ─── Sub-agent entry point ────────────────────────────────

    async def run_once(self, prompt: str) -> dict:
        self._output_buffer = [] # 步骤 1：激活输出重定向缓冲区
        before = self._usage_totals()
        before_calls = self._api_calls
        await self.chat(prompt)  # 步骤 2：直接复用主循环进行逻辑驱动
        text = "".join(self._output_buffer)  # 步骤 3：收集所有流式文本
        self._output_buffer = None  # 恢复默认打印模式
        after = self._usage_totals()
        return {
            "text": text,
            "tokens": {key: after[key] - before[key] for key in after},
            # Delivered as a delta, like the token counts: the parent adds it to
            # its own call count so "this round (N calls)" stays consistent with
            # the token sums it absorbed.
            "calls": self._api_calls - before_calls,
        }

    # ─── Output helper ────────────────────────────────────────

    def _emit_text(self, text: str) -> None:
        if self._output_buffer is not None:
            self._output_buffer.append(text)
        else:
            print_assistant_text(text)

    # ─── REPL commands ────────────────────────────────────────

    def clear_history(self) -> None:
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_reasoning_tokens = 0
        self.last_input_token_count = 0
        self._round_base = self._usage_totals()
        self._api_calls = 0
        self._round_api_base = 0
        print_info("Conversation cleared.")

    def switch_model(self, model: str) -> str:
        """Switch models without guessing their reasoning capability locally."""
        model = model.strip()
        if not model:
            raise ValueError("Model name cannot be empty")
        self.model = model
        self.context_window = _get_context_window(model)
        self.effective_window = self.context_window - 20000
        self._thinking_mode = self._resolve_thinking_mode()
        self.refresh_pricing()
        return self.model

    async def list_models(self) -> list[str]:
        """List model IDs exposed by the configured API backend."""
        page: Any
        if self.use_openai:
            assert self._openai_client is not None
            page = await self._openai_client.models.list()
        else:
            assert self._anthropic_client is not None
            page = await self._anthropic_client.models.list(limit=100)

        model_ids: set[str] = set()
        for item in getattr(page, "data", []) or []:
            model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if model_id:
                model_ids.add(str(model_id))
        if not model_ids:
            raise RuntimeError(
                f"The {self.backend} endpoint returned no models. "
                "Use /model <name> to switch manually."
            )
        model_ids.add(self.model)
        return sorted(model_ids, key=str.casefold)

    def has_conversation_history(self) -> bool:
        if self.use_openai:
            return any(message.get("role") != "system" for message in self._openai_messages)
        return bool(self._anthropic_messages)

    def show_cost(self) -> None:
        """``/cost`` — this round vs the session, with the pricing source."""
        resolution = self._price_resolution
        session = self._usage_totals()
        round_tokens = self._round_tokens()

        lines = [
            f"model     {self.model}  ({self.backend})",
            f"pricing   {resolution.source}",
            f"          {resolution.summary()}",
        ]
        if resolution.estimated:
            lines.append(
                "          estimated — no built-in price; override via "
                "DORO_PRICES in .env"
            )
        lines.append("")
        lines.append(
            "this round " + format_usage_line(
                round_tokens["input"],
                round_tokens["output"],
                round_tokens["cache_read"],
                round_tokens["cache_write"],
                round_tokens["reasoning"],
                self._cost_of(round_tokens),
                resolution.estimated,
                calls=self._round_calls(),
            )
        )
        lines.append(
            "session    " + format_usage_line(
                session["input"],
                session["output"],
                session["cache_read"],
                session["cache_write"],
                session["reasoning"],
                self._cost_of(session),
                resolution.estimated,
            )
        )
        fresh = session["input"] - session["cache_read"] - session["cache_write"]
        if session["cache_read"] and fresh <= 0:
            # Every backend-reported token is accounted for as cached read or
            # write, leaving nothing uncached. Real prompt caching cannot do that
            # for content the model has never seen, so this is a backend-side
            # accounting choice — say so instead of implying a verified hit rate.
            lines.append("")
            lines.append(
                "cache     backend reports no uncached input (0 fresh); the cached"
            )
            lines.append(
                "          share is its own figure, not a verifiable hit rate"
            )
        if self.max_cost_usd is not None:
            lines.append(
                f"budget    ${self._get_current_cost_usd():.4f} / ${self.max_cost_usd}"
            )
        if self.max_turns is not None:
            lines.append(f"turns     {self.current_turns}/{self.max_turns}")
        lines.append(
            f"context   {self.last_input_token_count} / {self.context_window} "
            f"(effective {self.effective_window})"
        )
        print_panel("Cost", lines)

    def _get_current_cost_usd(self) -> float:
        return self._cost_of(self._usage_totals())

    def _emit_turn_cost(self) -> None:
        """End-of-turn footer: this round's spend, one line, under the divider.

        The session totals still travel with the event (the legacy renderer
        falls back to them, and ``/cost`` reports both), but the footer only
        prints the round. Skipped when the round billed nothing — a Ctrl+C
        before the first response should not print a row of zeroes.
        """
        if self.is_sub_agent or self._round_calls() <= 0:
            return
        session = self._usage_totals()
        round_tokens = self._round_tokens()
        resolution = self._price_resolution
        round_usage: dict[str, Any] = dict(round_tokens)
        round_usage["cost"] = self._cost_of(round_tokens)
        round_usage["calls"] = self._round_calls()
        print_cost(
            session["input"],
            session["output"],
            session["cache_read"],
            session["cache_write"],
            self._cache_read_discount,
            reasoning_tokens=session["reasoning"],
            session_cost=self._cost_of(session),
            round_usage=round_usage,
            model=self.model,
            price_source=resolution.source,
            estimated=resolution.estimated,
        )

    def _check_budget(self) -> dict:
        if self.max_cost_usd is not None and self._get_current_cost_usd() >= self.max_cost_usd:
            return {"exceeded": True, "reason": f"Cost limit reached (${self._get_current_cost_usd():.4f} >= ${self.max_cost_usd})"}
        if self.max_turns is not None and self.current_turns >= self.max_turns:
            return {"exceeded": True, "reason": f"Turn limit reached ({self.current_turns} >= {self.max_turns})"}
        return {"exceeded": False}

    async def compact(self) -> None:
        await self._compact_conversation()

    # ─── Session ──────────────────────────────────────────────

    def restore_session(self, data: dict) -> None:
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Session metadata is invalid")
        stored_backend = metadata.get("backend")
        if stored_backend is None:
            if data.get("openaiMessages") is not None:
                stored_backend = "openai"
            elif data.get("anthropicMessages") is not None:
                stored_backend = "anthropic"
        if stored_backend and stored_backend != self.backend:
            raise ValueError(
                f"Session uses {stored_backend}, but the current backend is {self.backend}. "
                "Cross-backend resume is not supported."
            )

        messages_key = "openaiMessages" if self.use_openai else "anthropicMessages"
        messages = data.get(messages_key)
        if not isinstance(messages, list):
            raise ValueError(f"Session has no {self.backend} conversation history")

        previous_effort = self.reasoning_effort
        previous_explicit = self._reasoning_effort_explicit
        stored_effort = metadata.get("reasoningEffort")
        if stored_effort and not self._reasoning_effort_explicit:
            self.set_reasoning_effort(str(stored_effort), explicit=False)
        stored_model = metadata.get("model")
        try:
            if stored_model:
                self.switch_model(str(stored_model))
        except Exception:
            self.reasoning_effort = previous_effort
            self._reasoning_effort_explicit = previous_explicit
            self.thinking = previous_effort not in ("auto", "off")
            self._thinking_mode = self._resolve_thinking_mode()
            raise
        stored_id = metadata.get("id")
        if stored_id:
            self.session_id = str(stored_id)
        self.session_start_time = metadata.get("startTime") or self.session_start_time
        self._last_user_preview = str(metadata.get("preview") or "")
        if self.use_openai:
            self._openai_messages = messages
            self._anthropic_messages = []
        else:
            self._anthropic_messages = messages
            self._openai_messages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_reasoning_tokens = 0
        self.last_input_token_count = 0
        self._round_base = self._usage_totals()
        self._api_calls = 0
        self._round_api_base = 0
        self.current_turns = 0
        print_info(
            f"Session {self.session_id} restored "
            f"({self._get_message_count()} messages, model: {self.model})."
        )

    def _get_message_count(self) -> int:
        return len(self._openai_messages) if self.use_openai else len(self._anthropic_messages)

    def _auto_save(self) -> None:
        try:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save_session(self.session_id, {
                "metadata": {
                    "id": self.session_id,
                    "model": self.model,
                    "backend": self.backend,
                    "reasoningEffort": self.reasoning_effort,
                    "cwd": str(Path.cwd()),
                    "startTime": self.session_start_time,
                    "updatedAt": now,
                    "messageCount": self._get_message_count(),
                    "preview": self._last_user_preview,
                    "schemaVersion": 3,
                },
                "anthropicMessages": self._anthropic_messages if not self.use_openai else None,
                "openaiMessages": self._openai_messages if self.use_openai else None,
            })
        except Exception:
            pass

    # ─── Autocompact ──────────────────────────────────────────

    async def _check_and_compact(self) -> None:
        if self.last_input_token_count > self.effective_window * 0.85:
            print_info("Context window filling up, compacting conversation...")
            await self._compact_conversation()

    async def _compact_conversation(self) -> None:
        if self.use_openai:
            await self._compact_openai()
        else:
            await self._compact_anthropic()
        print_info("Conversation compacted.")

    async def _compact_anthropic(self) -> None:
        # Invariant: caller must ensure the last message is a plain user-text
        # message (not a tool_result). We slice it off below; if it were a
        # tool_result, the preceding assistant's tool_use would be orphaned
        # and the API would reject the summarize call.
        if len(self._anthropic_messages) < 4:
            return
        assert self._anthropic_client is not None
        last_user_msg = self._anthropic_messages[-1]
        summary_resp = await self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=2048,
            system="You are a conversation summarizer. Be concise but preserve important details.",
            messages=cast(
                list[anthropic.types.MessageParam],
                [
                    *self._anthropic_messages[:-1],
                    {
                        "role": "user",
                        "content": (
                            "Summarize the conversation so far in a concise paragraph, "
                            "preserving key decisions, file paths, and context needed "
                            "to continue the work."
                        ),
                    },
                ],
            ),
        )
        first_block = summary_resp.content[0] if summary_resp.content else None
        summary_text = (
            first_block.text
            if isinstance(first_block, anthropic.types.TextBlock)
            else "No summary available."
        )
        # The summarizer re-reads the whole conversation, so it is the single
        # most expensive call in a session. Booking it here keeps `--max-cost`
        # honest. `last_input_token_count` is left to the reset below: the
        # history is being replaced, so the old gauge no longer describes it.
        if getattr(summary_resp, "usage", None) is not None:
            comp_in, comp_write, comp_read = _anthropic_usage_totals(summary_resp.usage)
            self.total_input_tokens += comp_in
            self.total_output_tokens += summary_resp.usage.output_tokens
            self.total_cache_write_tokens += comp_write
            self.total_cache_read_tokens += comp_read
            self._api_calls += 1
        self._anthropic_messages = [
            {"role": "user", "content": f"[Previous conversation summary]\n{summary_text}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
        ]
        if last_user_msg.get("role") == "user":
            self._anthropic_messages.append(last_user_msg)
        self.last_input_token_count = 0

    async def _compact_openai(self) -> None:
        # Invariant: caller must ensure the last message is a plain user-text
        # message (not a `tool` role result). Same reasoning as
        # _compact_anthropic — slicing off a tool result would orphan the
        # preceding assistant's tool_calls.
        if len(self._openai_messages) < 5:
            return
        assert self._openai_client is not None
        system_msg = self._openai_messages[0]
        last_user_msg = self._openai_messages[-1]
        summary_resp = await self._openai_client.chat.completions.create(
            model=self.model,
            messages=cast(
                list[openai.types.chat.ChatCompletionMessageParam],
                [
                    {"role": "system", "content": "You are a conversation summarizer. Be concise but preserve important details."},
                    *self._openai_messages[1:-1],
                    {"role": "user", "content": "Summarize the conversation so far in a concise paragraph, preserving key decisions, file paths, and context needed to continue the work."},
                ],
            ),
        )
        summary_text = summary_resp.choices[0].message.content or "No summary available."
        # Same reasoning as _compact_anthropic: a full-history summarize call is
        # the biggest single spend of a session and used to go unbooked.
        if getattr(summary_resp, "usage", None) is not None:
            comp_in, comp_out, comp_read = _openai_usage_totals(summary_resp.usage)
            self.total_input_tokens += comp_in
            self.total_output_tokens += comp_out
            self.total_cache_read_tokens += comp_read
            self.total_reasoning_tokens += _openai_reasoning_tokens(summary_resp.usage)
            self._api_calls += 1
        self._openai_messages = [
            system_msg,
            {"role": "user", "content": f"[Previous conversation summary]\n{summary_text}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?"},
        ]
        if last_user_msg.get("role") == "user":
            self._openai_messages.append(last_user_msg)
        self.last_input_token_count = 0

    # ─── Multi-tier compression pipeline ──────────────────────

    def _run_compression_pipeline(self) -> None:
        if self.use_openai:
            self._budget_tool_results_openai()
            self._snip_stale_results_openai()
            self._microcompact_openai()
        else:
            self._budget_tool_results_anthropic()
            self._snip_stale_results_anthropic()
            self._microcompact_anthropic()

    # Tier 1: Budget tool results
    def _budget_tool_results_anthropic(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < 0.5:
            return
        budget = 15000 if utilization > 0.7 else 30000
        for msg in self._anthropic_messages:
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and len(block["content"]) > budget:
                    keep = (budget - 80) // 2
                    block["content"] = block["content"][:keep] + f"\n\n[... budgeted: {len(block['content']) - keep * 2} chars truncated ...]\n\n" + block["content"][-keep:]

    def _budget_tool_results_openai(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < 0.5:
            return
        budget = 15000 if utilization > 0.7 else 30000
        for msg in self._openai_messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > budget:
                keep = (budget - 80) // 2
                msg["content"] = msg["content"][:keep] + f"\n\n[... budgeted: {len(msg['content']) - keep * 2} chars truncated ...]\n\n" + msg["content"][-keep:]

    # Tier 2: Snip stale results
    def _snip_stale_results_anthropic(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return

        results = []
        for mi, msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] != SNIP_PLACEHOLDER:
                    tool_use_id = block.get("tool_use_id")
                    tool_info = self._find_tool_use_by_id(tool_use_id)
                    if tool_info and tool_info["name"] in SNIPPABLE_TOOLS:
                        results.append({"mi": mi, "bi": bi, "name": tool_info["name"], "file_path": tool_info.get("input", {}).get("file_path")})

        if len(results) <= KEEP_RECENT_RESULTS:
            return

        to_snip = set()
        seen_files: dict[str, list[int]] = {}
        for i, r in enumerate(results):
            if r["name"] == "read_file" and r.get("file_path"):
                seen_files.setdefault(r["file_path"], []).append(i)

        for indices in seen_files.values():
            if len(indices) > 1:
                for j in indices[:-1]:
                    to_snip.add(j)

        snip_before = len(results) - KEEP_RECENT_RESULTS
        for i in range(snip_before):
            to_snip.add(i)

        for idx in to_snip:
            r = results[idx]
            self._anthropic_messages[r["mi"]]["content"][r["bi"]]["content"] = SNIP_PLACEHOLDER

    def _snip_stale_results_openai(self) -> None:
        utilization = self.last_input_token_count / self.effective_window if self.effective_window else 0
        if utilization < SNIP_THRESHOLD:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] != SNIP_PLACEHOLDER:
                tool_msgs.append(i)
        if len(tool_msgs) <= KEEP_RECENT_RESULTS:
            return
        snip_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(snip_count):
            self._openai_messages[tool_msgs[i]]["content"] = SNIP_PLACEHOLDER

    # Tier 3: Microcompact
    def _microcompact_anthropic(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        all_results = []
        for mi, msg in enumerate(self._anthropic_messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            for bi, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str) and block["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                    all_results.append((mi, bi))
        clear_count = len(all_results) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            mi, bi = all_results[i]
            self._anthropic_messages[mi]["content"][bi]["content"] = "[Old result cleared]"

    def _microcompact_openai(self) -> None:
        if not self.last_api_call_time or (time.time() - self.last_api_call_time) < MICROCOMPACT_IDLE_S:
            return
        tool_msgs = []
        for i, msg in enumerate(self._openai_messages):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and msg["content"] not in (SNIP_PLACEHOLDER, "[Old result cleared]"):
                tool_msgs.append(i)
        clear_count = len(tool_msgs) - KEEP_RECENT_RESULTS
        for i in range(max(0, clear_count)):
            self._openai_messages[tool_msgs[i]]["content"] = "[Old result cleared]"

    def _find_tool_use_by_id(self, tool_use_id: str | None) -> dict | None:
        """Find a ``tool_use`` block by id. ``None`` simply never matches.

        A ``tool_result`` block is not guaranteed to carry ``tool_use_id``, so
        callers pass whatever they found instead of pre-filtering.
        """
        for msg in self._anthropic_messages:
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return {"name": block["name"], "input": block.get("input", {})}
        return None

    # ─── Large result persistence ─────────────────────────────────
    # When a tool result exceeds 30 KB, write it to disk and replace the
    # context entry with a short preview + file path.  The model can use
    # read_file to retrieve the full output later — no information is lost.

    def _persist_large_result(self, tool_name: str, result: str) -> str:
        THRESHOLD = 30 * 1024  # 30 KB
        if len(result.encode()) <= THRESHOLD:
            return result
        d = Path.home() / ".doro" / "tool-results"
        d.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time() * 1000)}-{tool_name}.txt"
        filepath = d / filename
        filepath.write_text(result, encoding="utf-8")

        lines = result.split("\n")
        preview = "\n".join(lines[:200])
        size_kb = len(result.encode()) / 1024

        return (
            f"[Result too large ({size_kb:.1f} KB, {len(lines)} lines). "
            f"Full output saved to {filepath}. "
            f"You can use read_file to see the full result.]\n\n"
            f"Preview (first 200 lines):\n{preview}"
        )

    # ─── Execute tool (handles agent/skill/plan mode internally) ─────

    def _is_concurrency_safe(self, name: str) -> bool:
        """Whether a tool may start executing while the model is still streaming.

        Built-in read-only tools always qualify. MCP tools qualify only when
        their server is declared "readOnly": true in .mcp.json — an MCP tool can
        have arbitrary side effects, so parallelism there is opt-in.
        """
        if name in CONCURRENCY_SAFE_TOOLS:
            return True
        return self._mcp_manager.is_concurrency_safe(name)

    def _mcp_read_only_tool(self, name: str) -> bool:
        """Whether an MCP tool's server is declared read-only (plan-mode gate).

        Handed to ``check_permission``, which denies MCP tools in plan mode
        unless the server opted in via ``"readOnly": true``. Any name that is
        not an MCP tool answers False, so the gate stays fail-closed even if it
        is reached with something unexpected.
        """
        return self._mcp_manager.is_read_only_tool(name)

    async def _execute_tool_call(self, name: str, inp: dict) -> str:
        if name in ("enter_plan_mode", "exit_plan_mode"):
            return await self._execute_plan_mode_tool(name)
        if name == "agent":
            return await self._execute_agent_tool(inp)
        if name == "skill":
            return await self._execute_skill_tool(inp)
        # Route MCP tool calls to the MCP manager
        if self._mcp_manager.is_mcp_tool(name):
            return await self._mcp_manager.call_tool(name, inp)
        return await execute_tool(name, inp, self._read_file_state)

    # ─── Skill fork mode ─────────────────────────────────────

    async def _execute_skill_tool(self, inp: dict) -> str:
        from .skills import execute_skill
        result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))
        if not result:
            return f"Unknown skill: {inp.get('skill_name', '')}"

        if result["context"] == "fork":
            tools = (
                [t for t in self.tools if t["name"] in result["allowed_tools"]]
                if result.get("allowed_tools")
                else [t for t in self.tools if t["name"] != "agent"]
            )
            print_sub_agent_start("skill-fork", inp.get("skill_name", ""))
            sub_agent = Agent(
                model=self.model,
                backend=self.backend,
                api_base=self.api_base,
                anthropic_base_url=self.anthropic_base_url,
                api_key=self.api_key,
                reasoning_effort=self.reasoning_effort,
                default_reasoning_effort=self.default_reasoning_effort,
                custom_system_prompt=result["prompt"],
                custom_tools=tools,
                is_sub_agent=True,
                permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
            )
            try:
                sub_result = await sub_agent.run_once(inp.get("args") or "Execute this skill task.")
                self.total_input_tokens += sub_result["tokens"]["input"]
                self.total_output_tokens += sub_result["tokens"]["output"]
                self.total_cache_read_tokens += sub_result["tokens"].get("cache_read", 0)
                self.total_cache_write_tokens += sub_result["tokens"].get("cache_write", 0)
                self.total_reasoning_tokens += sub_result["tokens"].get("reasoning", 0)
                self._api_calls += sub_result.get("calls", 0)
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return sub_result["text"] or "(Skill produced no output)"
            except Exception as e:
                print_sub_agent_end("skill-fork", inp.get("skill_name", ""))
                return f"Skill fork error: {e}"

        return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

    # ─── Plan mode helpers ──────────────────────────────────────

    def _generate_plan_file_path(self) -> str:
        d = Path.home() / ".claude" / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"plan-{self.session_id}.md")

    def _build_plan_mode_prompt(self) -> str:
        return f"""

# Plan Mode Active

Plan mode is active. You MUST NOT make any edits (except the plan file below), run non-readonly tools, or make any changes to the system.

## Plan File: {self._plan_file_path}
Write your plan incrementally to this file using write_file or edit_file. This is the ONLY file you are allowed to edit.

## Workflow
1. **Explore**: Read code to understand the task. Use read_file, list_files, grep_search.
2. **Design**: Design your implementation approach. Use the agent tool with type="plan" if the task is complex.
3. **Write Plan**: Write a structured plan to the plan file including:
   - **Context**: Why this change is needed
   - **Steps**: Implementation steps with critical file paths
   - **Verification**: How to test the changes
4. **Exit**: Call exit_plan_mode when your plan is ready for user review.

IMPORTANT: When your plan is complete, you MUST call exit_plan_mode. Do NOT ask the user to approve — exit_plan_mode handles that."""

    async def _execute_plan_mode_tool(self, name: str) -> str:
        if name == "enter_plan_mode":
            if self.permission_mode == "plan":
                return "Already in plan mode."
            self._pre_plan_mode = self.permission_mode
            self.permission_mode = "plan"
            self._plan_file_path = self._generate_plan_file_path()
            self._system_prompt = self._base_system_prompt + self._build_plan_mode_prompt()
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Entered plan mode (read-only). Plan file: " + self._plan_file_path)
            return f"Entered plan mode. You are now in read-only mode.\n\nYour plan file: {self._plan_file_path}\nWrite your plan to this file. This is the only file you can edit.\n\nWhen your plan is complete, call exit_plan_mode."

        if name == "exit_plan_mode":
            if self.permission_mode != "plan":
                return "Not in plan mode."
            plan_content = "(No plan file found)"
            if self._plan_file_path and Path(self._plan_file_path).exists():
                plan_content = Path(self._plan_file_path).read_text(encoding="utf-8")

            # Interactive approval flow
            if self._plan_approval_fn:
                result = await self._plan_approval_fn(plan_content)
                choice = result.get("choice", "manual-execute")

                if choice == "keep-planning":
                    feedback = result.get("feedback") or "Please revise the plan."
                    return (
                        f"User rejected the plan and wants to keep planning.\n\n"
                        f"User feedback: {feedback}\n\n"
                        f"Please revise your plan based on this feedback. When done, call exit_plan_mode again."
                    )

                # User approved — determine target mode
                if choice == "clear-and-execute":
                    target_mode = "acceptEdits"
                elif choice == "execute":
                    target_mode = "acceptEdits"
                else:  # manual-execute
                    target_mode = self._pre_plan_mode or "default"

                # Exit plan mode
                self.permission_mode = target_mode
                self._pre_plan_mode = None
                saved_plan_path = self._plan_file_path
                self._plan_file_path = None
                self._system_prompt = self._base_system_prompt
                if self.use_openai and self._openai_messages:
                    self._openai_messages[0]["content"] = self._system_prompt

                if choice == "clear-and-execute":
                    self._clear_history_keep_system()
                    self._context_cleared = True
                    print_info(f"Plan approved. Context cleared, executing in {target_mode} mode.")
                    return (
                        f"User approved the plan. Context was cleared. Permission mode: {target_mode}\n\n"
                        f"Plan file: {saved_plan_path}\n\n"
                        f"## Approved Plan:\n{plan_content}\n\n"
                        f"Proceed with implementation."
                    )

                print_info(f"Plan approved. Executing in {target_mode} mode.")
                return (
                    f"User approved the plan. Permission mode: {target_mode}\n\n"
                    f"## Approved Plan:\n{plan_content}\n\n"
                    f"Proceed with implementation."
                )

            # Fallback: no approval function (e.g. sub-agents)
            self.permission_mode = self._pre_plan_mode or "default"
            self._pre_plan_mode = None
            self._plan_file_path = None
            self._system_prompt = self._base_system_prompt
            if self.use_openai and self._openai_messages:
                self._openai_messages[0]["content"] = self._system_prompt
            print_info("Exited plan mode. Restored to " + self.permission_mode + " mode.")
            return f"Exited plan mode. Permission mode restored to: {self.permission_mode}\n\n## Your Plan:\n{plan_content}"

        return f"Unknown plan mode tool: {name}"

    def _clear_history_keep_system(self) -> None:
        """Clear history but keep system prompt (used for clear-context plan approval)."""
        self._anthropic_messages = []
        self._openai_messages = []
        if self.use_openai:
            self._openai_messages.append({"role": "system", "content": self._system_prompt})
        self.last_input_token_count = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_reasoning_tokens = 0

    async def _execute_agent_tool(self, inp: dict) -> str:
        agent_type = inp.get("type", "general")
        description = inp.get("description", "sub-agent task")
        prompt = inp.get("prompt", "")

        print_sub_agent_start(agent_type, description)

        config = get_sub_agent_config(agent_type)
        sub_agent = Agent(
            model=self.model,
            backend=self.backend,
            api_base=self.api_base,
            anthropic_base_url=self.anthropic_base_url,
            api_key=self.api_key,
            reasoning_effort=self.reasoning_effort,
            default_reasoning_effort=self.default_reasoning_effort,
            custom_system_prompt=config["system_prompt"],
            custom_tools=config["tools"],
            is_sub_agent=True,
            permission_mode="plan" if self.permission_mode == "plan" else "bypassPermissions",
        )

        try:
            result = await sub_agent.run_once(prompt)
            self.total_input_tokens += result["tokens"]["input"]
            self.total_output_tokens += result["tokens"]["output"]
            self.total_cache_read_tokens += result["tokens"].get("cache_read", 0)
            self.total_cache_write_tokens += result["tokens"].get("cache_write", 0)
            self.total_reasoning_tokens += result["tokens"].get("reasoning", 0)
            self._api_calls += result.get("calls", 0)
            print_sub_agent_end(agent_type, description)
            return result["text"] or "(Sub-agent produced no output)"
        except Exception as e:
            print_sub_agent_end(agent_type, description)
            return f"Sub-agent error: {e}"

    # ─── Anthropic backend ───────────────────────────────────────

    async def _chat_anthropic(self, user_message: str) -> None:
        assert self._anthropic_client is not None, "anthropic backend without a client"
        self._anthropic_messages.append({"role": "user", "content": user_message})
        # Auto-compact at turn boundary only — the last message is now plain
        # user text, so the slice in _compact_anthropic won't sever a
        # tool_use ↔ tool_result pair from the previous turn's tool execution.
        await self._check_and_compact()

        # Start async memory prefetch (non-blocking, fires once per user turn)
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()

            # Consume memory prefetch if settled (non-blocking poll, zero-wait).
            # Append to last user message to maintain user/assistant alternation.
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        last = self._anthropic_messages[-1] if self._anthropic_messages else None
                        if last and last.get("role") == "user":
                            content = last.get("content", "")
                            if isinstance(content, str):
                                last["content"] = content + "\n\n" + injection_text
                            elif isinstance(content, list):
                                content.append({"type": "text", "text": injection_text})
                        else:
                            self._anthropic_messages.append({"role": "user", "content": injection_text})
                        for m in memories:
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += len(m.content.encode())
                except Exception:
                    pass  # prefetch errors already logged

            if not self.is_sub_agent:
                start_spinner()

            # ── Streaming tool execution ──────────────────────────────
            # As each tool_use content block completes during streaming, check
            # if it's concurrency-safe and auto-allowed. If so, start execution
            # immediately — the tool runs while the model still generates.
            early_executions: dict[str, asyncio.Task] = {}

            def _on_tool_block(block: dict):
                if self._is_concurrency_safe(block["name"]):
                    perm = check_permission(
                        block["name"],
                        block["input"],
                        self.permission_mode,
                        self._plan_file_path,
                        mcp_read_only=self._mcp_read_only_tool,
                    )
                    if perm["action"] == "allow":
                        task = asyncio.create_task(self._execute_tool_call(block["name"], block["input"]))
                        # B023 warns that this closure does not bind the loop
                        # variable — true in general, not here: the dict and this
                        # function are rebuilt together every iteration and both
                        # are consumed inside the single `_call_anthropic_stream`
                        # await below, so it can never see a later iteration.
                        early_executions[block["id"]] = task  # noqa: B023

            response = await self._call_anthropic_stream(on_tool_block_complete=_on_tool_block)

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()
            input_tokens, cache_write, cache_read = _anthropic_usage_totals(response.usage)
            self._api_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_cache_write_tokens += cache_write
            self.total_cache_read_tokens += cache_read
            self.last_input_token_count = input_tokens

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            self._anthropic_messages.append({
                "role": "assistant",
                "content": [self._block_to_dict(b) for b in response.content],
            })

            if not tool_uses:
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                break

            # Process tools: early-started ones (from streaming) just await
            # their result; others go through permission check + execution.
            tool_results: list[dict] = []
            context_break = False
            for tu in tool_uses:
                if context_break or self._aborted:
                    break
                inp = dict(tu.input) if hasattr(tu.input, 'items') else tu.input
                print_tool_call(tu.name, inp)

                # Was this tool already started during streaming?
                early_task = early_executions.get(tu.id)
                if early_task:
                    raw = await early_task
                    res = self._persist_large_result(tu.name, raw)
                    print_tool_result(tu.name, res)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                    continue

                # Permission check for tools not started early
                perm = check_permission(
                    tu.name,
                    inp,
                    self.permission_mode,
                    self._plan_file_path,
                    mcp_read_only=self._mcp_read_only_tool,
                )
                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])

                raw = await self._execute_tool_call(tu.name, inp)
                res = self._persist_large_result(tu.name, raw)
                print_tool_result(tu.name, res)

                if self._context_cleared:
                    self._context_cleared = False
                    self._anthropic_messages.append({"role": "user", "content": res})
                    context_break = True
                    break
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})

            if not context_break and tool_results:
                self._anthropic_messages.append({"role": "user", "content": tool_results})
            self._context_cleared = False

    @staticmethod
    def _block_to_dict(block) -> dict:
        """Convert an Anthropic content block to a plain dict for storage."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input) if hasattr(block.input, 'items') else block.input}
        # Fallback
        return {"type": block.type}

    async def _call_anthropic_stream(self, on_tool_block_complete=None):
        """Stream an Anthropic API call. When a tool_use content block finishes
        during streaming, on_tool_block_complete fires immediately so the caller
        can start execution before the full response arrives (streaming tool
        execution -- mirrors Claude Code's content_block_stop approach)."""
        client = self._anthropic_client
        assert client is not None, "anthropic backend without a client"

        async def _do(stream_state: dict[str, bool]):
            max_output = _get_max_output_tokens(self.model)
            create_params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_output if self._thinking_mode == "adaptive" else 16384,
                "system": self._system_prompt,
                "tools": get_active_tool_definitions(self.tools),
                "messages": self._anthropic_messages,
            }
            create_params.update(self._anthropic_reasoning_params())

            first_text = True
            # Track in-flight tool_use blocks by index for streaming execution
            tool_blocks_by_index: dict[int, dict] = {}

            async with client.messages.stream(**create_params) as stream:
                async for event in stream:
                    if not hasattr(event, 'type'):
                        continue

                    if event.type == "content_block_start":
                        cb = getattr(event, 'content_block', None)
                        if cb is not None:
                            stream_state["started"] = True
                        if cb and getattr(cb, 'type', None) == "tool_use":
                            tool_blocks_by_index[event.index] = {
                                "id": cb.id, "name": cb.name, "input_json": "",
                            }

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, 'text'):
                            stream_state["started"] = True
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n")
                                first_text = False
                            self._emit_text(delta.text)
                        elif hasattr(delta, 'thinking'):
                            stream_state["started"] = True
                            if first_text:
                                stop_spinner()
                                self._emit_text("\n  [thinking] ")
                                first_text = False
                            self._emit_text(delta.thinking)
                        elif hasattr(delta, 'partial_json'):
                            stream_state["started"] = True
                            tb = tool_blocks_by_index.get(event.index)
                            if tb:
                                tb["input_json"] += delta.partial_json

                    elif event.type == "content_block_stop":
                        tb = tool_blocks_by_index.pop(event.index, None)
                        if tb and on_tool_block_complete:
                            import json as _json
                            try:
                                parsed = _json.loads(tb["input_json"] or "{}")
                            except Exception:
                                parsed = {}
                            on_tool_block_complete({
                                "type": "tool_use", "id": tb["id"],
                                "name": tb["name"], "input": parsed,
                            })

                final_message = await stream.get_final_message()

            # Filter out thinking blocks
            final_message.content = [b for b in final_message.content if b.type != "thinking"]
            return final_message

        return await self._with_reasoning_fallback(_do)

    # ─── OpenAI-compatible backend ───────────────────────────────

    async def _chat_openai(self, user_message: str) -> None:
        assert self._openai_client is not None, "openai backend without a client"
        self._openai_messages.append({"role": "user", "content": user_message})
        # Auto-compact at turn boundary only — see _chat_anthropic for rationale.
        # The last message is now plain user text, so the slice in
        # _compact_openai won't orphan a tool_calls / tool message pair.
        await self._check_and_compact()

        # Start async memory prefetch (non-blocking, fires once per user turn)
        memory_prefetch: MemoryPrefetch | None = None
        if not self.is_sub_agent:
            sq = self._build_side_query()
            if sq:
                memory_prefetch = start_memory_prefetch(
                    user_message, sq,
                    self._already_surfaced_memories, self._session_memory_bytes,
                )

        while True:
            if self._aborted:
                break

            self._run_compression_pipeline()

            # Consume memory prefetch if settled (non-blocking poll, zero-wait)
            if memory_prefetch and memory_prefetch.settled and not memory_prefetch.consumed:
                memory_prefetch.consumed = True
                try:
                    memories = memory_prefetch.task.result()
                    if memories:
                        injection_text = format_memories_for_injection(memories)
                        last = self._openai_messages[-1] if self._openai_messages else None
                        if last and last.get("role") == "user":
                            last["content"] = (last.get("content") or "") + "\n\n" + injection_text
                        else:
                            self._openai_messages.append({"role": "user", "content": injection_text})
                        for m in memories:
                            self._already_surfaced_memories.add(m.path)
                            self._session_memory_bytes += len(m.content.encode())
                except Exception:
                    pass  # prefetch errors already logged

            if not self.is_sub_agent:
                start_spinner()

            response = await self._call_openai_stream()

            if not self.is_sub_agent:
                stop_spinner()

            self.last_api_call_time = time.time()

            usage = response.get("usage")
            if usage is not None:
                input_tokens, output_tokens, cache_read = _openai_usage_totals(usage)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cache_read_tokens += cache_read
                # Reasoning tokens are a *subset* of completion_tokens — shown
                # for visibility, never added to the output total.
                self.total_reasoning_tokens += _openai_reasoning_tokens(usage)
                self._api_calls += 1
                # `prompt_tokens` is already the full prompt, cached prefix
                # included — this is the one line that differs from the
                # Anthropic branch, where the cache read has to be added in.
                self.last_input_token_count = input_tokens

            choice = response.get("choices", [{}])[0] if response.get("choices") else {}
            message = choice.get("message", {})

            self._openai_messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                break

            self.current_turns += 1
            budget = self._check_budget()
            if budget["exceeded"]:
                print_info(f"Budget exceeded: {budget['reason']}")
                break

            # Phase 1: Parse & permission-check (serial)
            oai_checked: list[dict] = []
            for tc in tool_calls:
                if self._aborted:
                    break
                if tc.get("type") != "function":
                    continue
                fn_name = tc["function"]["name"]
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except Exception:
                    inp = {}

                print_tool_call(fn_name, inp)

                perm = check_permission(
                    fn_name,
                    inp,
                    self.permission_mode,
                    self._plan_file_path,
                    mcp_read_only=self._mcp_read_only_tool,
                )
                if perm["action"] == "deny":
                    print_info(f"Denied: {perm.get('message', '')}")
                    oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False, "result": f"Action denied: {perm.get('message', '')}"})
                    continue
                if perm["action"] == "confirm" and perm.get("message") and perm["message"] not in self._confirmed_paths:
                    confirmed = await self._confirm_dangerous(perm["message"])
                    if not confirmed:
                        oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": False, "result": "User denied this action."})
                        continue
                    self._confirmed_paths.add(perm["message"])
                oai_checked.append({"tc": tc, "fn": fn_name, "inp": inp, "allowed": True})

            # Phase 2: Group & execute (parallel for consecutive safe tools)
            oai_batches: list[dict] = []
            for ct in oai_checked:
                safe = ct["allowed"] and ct["fn"] in CONCURRENCY_SAFE_TOOLS
                if safe and oai_batches and oai_batches[-1]["concurrent"]:
                    oai_batches[-1]["items"].append(ct)
                else:
                    oai_batches.append({"concurrent": safe, "items": [ct]})

            oai_context_break = False
            for batch in oai_batches:
                if oai_context_break or self._aborted:
                    break

                if batch["concurrent"]:
                    async def _run_oai_safe(ct_item: dict) -> tuple[dict, str]:
                        raw = await self._execute_tool_call(ct_item["fn"], ct_item["inp"])
                        res = self._persist_large_result(ct_item["fn"], raw)
                        print_tool_result(ct_item["fn"], res)
                        return ct_item, res

                    results = await asyncio.gather(*[_run_oai_safe(ct) for ct in batch["items"]])
                    for ct_item, res in results:
                        self._openai_messages.append({"role": "tool", "tool_call_id": ct_item["tc"]["id"], "content": res})
                else:
                    for ct in batch["items"]:
                        if not ct["allowed"]:
                            self._openai_messages.append({"role": "tool", "tool_call_id": ct["tc"]["id"], "content": ct["result"]})
                            continue
                        raw = await self._execute_tool_call(ct["fn"], ct["inp"])
                        res = self._persist_large_result(ct["fn"], raw)
                        print_tool_result(ct["fn"], res)

                        if self._context_cleared:
                            self._context_cleared = False
                            self._openai_messages.append({"role": "user", "content": res})
                            oai_context_break = True
                            break
                        self._openai_messages.append({"role": "tool", "tool_call_id": ct["tc"]["id"], "content": res})

            self._context_cleared = False

    async def _call_openai_stream(self) -> dict:
        client = self._openai_client
        assert client is not None, "openai backend without a client"

        async def _do(stream_state: dict[str, bool]):
            create_params: dict[str, Any] = {
                "model": self.model,
                "tools": _to_openai_tools(get_active_tool_definitions(self.tools)),
                "messages": self._openai_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            create_params.update(self._openai_reasoning_params())
            stream = await client.chat.completions.create(**create_params)

            content = ""
            first_text = True
            tool_calls: dict[int, dict] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                # Keep the raw SDK usage object: the cached-prompt count lives in
                # a nested details object that a flat dict copy would drop.
                if chunk.usage is not None:
                    usage = chunk.usage

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta and (
                    getattr(delta, "reasoning", None)
                    or getattr(delta, "reasoning_content", None)
                ):
                    stream_state["started"] = True

                if delta and delta.content:
                    stream_state["started"] = True
                    if first_text:
                        stop_spinner()
                        self._emit_text("\n")
                        first_text = False
                    self._emit_text(delta.content)
                    content += delta.content

                if delta and delta.tool_calls:
                    stream_state["started"] = True
                    for tc in delta.tool_calls:
                        existing = tool_calls.get(tc.index)
                        if existing:
                            if tc.function and tc.function.arguments:
                                existing["arguments"] += tc.function.arguments
                        else:
                            tool_calls[tc.index] = {
                                "id": tc.id or "",
                                "name": (tc.function.name if tc.function else "") or "",
                                "arguments": (tc.function.arguments if tc.function else "") or "",
                            }

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            assembled = None
            if tool_calls:
                assembled = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for _, tc in sorted(tool_calls.items())
                ]

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": assembled,
                    },
                    "finish_reason": finish_reason or "stop",
                }],
                # None, not zeros: a gateway that ignores `include_usage` used to
                # hand back 0/0, which overwrote `last_input_token_count` with 0
                # and silently froze auto-compaction. Better to book nothing.
                "usage": usage,
            }

        return await self._with_reasoning_fallback(_do)

    # ─── Shared ──────────────────────────────────────────────────

    async def _confirm_dangerous(self, command: str) -> bool:
        print_confirmation(command)
        if self.confirm_fn:
            return await self.confirm_fn(command)
        # Fallback: blocking input
        try:
            answer = input("  Allow? (y/n): ")
            return answer.lower().startswith("y")
        except EOFError:
            return False
