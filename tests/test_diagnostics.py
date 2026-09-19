"""``/config`` and ``/doctor``: what is in effect, and what is broken.

Two things these tests pin down: the reports never leak an API key, and they
say "estimated" out loud when the model's price is unknown.
"""

from __future__ import annotations

from unittest.mock import patch

from doro.agent import Agent, _openai_reasoning_tokens
from doro.diagnostics import (
    build_config_report,
    build_doctor_report,
    mask_secret,
    run_checks,
)

SECRET = "sk-supersecret-value-1234567890"


def _agent(**kwargs) -> Agent:
    with patch("doro.agent.openai.AsyncOpenAI"):
        return Agent(
            backend="openai",
            api_key=SECRET,
            model_source="--model",
            effort_source="--effort",
            **kwargs,
        )


def _isolate_mcp_config(monkeypatch, tmp_path) -> None:
    """Point both config readers at a directory with no config files."""
    empty = [tmp_path / ".mcp.json"]
    monkeypatch.setattr(
        "doro.mcp_client.config_search_paths", lambda cwd=None: list(empty)
    )
    monkeypatch.setattr(
        "doro.diagnostics.config_search_paths", lambda cwd=None: list(empty)
    )


# ─── masking ────────────────────────────────────────────────


def test_mask_keeps_a_hint_but_not_the_key() -> None:
    masked = mask_secret(SECRET)

    assert SECRET not in masked
    assert masked.startswith("sk-s")
    assert SECRET[-4:] in masked
    assert "not set" in mask_secret(None)


def test_short_secrets_are_fully_hidden() -> None:
    assert "abc" not in mask_secret("abc12345")


# ─── /config ────────────────────────────────────────────────


def test_config_report_never_contains_the_raw_key(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()

    title, lines = build_config_report(agent)
    rendered = "\n".join(lines)

    assert title == "Configuration"
    assert SECRET not in rendered
    assert "sk-s" in rendered  # the masked hint is present


def test_config_report_shows_actual_effort_without_requested_mode(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()

    _title, lines = build_config_report(agent)
    rendered = "\n".join(lines)

    assert "[--model]" in rendered
    assert "effort       medium" in rendered
    assert "[--effort]" not in rendered
    assert "thinking:" not in rendered
    assert "deepseek-v4.1-flash" in rendered
    assert "estimated" in rendered  # unknown gateway price is disclosed
    assert "python" in rendered


def test_config_report_lists_every_config_path_even_when_missing(
    monkeypatch, tmp_path
) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()

    _title, lines = build_config_report(agent)
    rendered = "\n".join(lines)

    assert "[missing]" in rendered
    assert str(tmp_path / ".mcp.json") in rendered


# ─── /doctor ────────────────────────────────────────────────


def _status_of(checks, label: str) -> str:
    return next(check.status for check in checks if check.label == label)


def test_doctor_fails_without_an_api_key(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    with patch("doro.agent.openai.AsyncOpenAI"):
        agent = Agent(backend="openai", api_key=None)

    checks = run_checks(agent)

    assert _status_of(checks, "api key") == "fail"
    assert _status_of(checks, "python") == "ok"


def test_doctor_warns_when_the_model_price_is_unknown(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)

    checks = run_checks(_agent())

    assert _status_of(checks, "pricing") == "warn"


def test_doctor_reports_a_known_model_price_as_ok(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)

    checks = run_checks(_agent(model="gpt-4o"))

    assert _status_of(checks, "pricing") == "ok"


def test_doctor_warns_when_no_mcp_servers_are_configured(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)

    checks = run_checks(_agent())

    assert _status_of(checks, "mcp config") == "warn"


def test_doctor_summary_counts_cover_every_check(monkeypatch, tmp_path) -> None:
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()

    checks = run_checks(agent)
    title, lines = build_doctor_report(agent)

    assert title == "Doctor"
    assert len(lines) == len(checks) + 2  # blank spacer + summary
    counts = {name: sum(1 for c in checks if c.status == name) for name in ("ok", "warn", "fail")}
    assert sum(counts.values()) == len(checks)
    assert f"{counts['ok']} ok" in lines[-1]


def test_doctor_never_raises_on_a_partial_agent(monkeypatch, tmp_path) -> None:
    """Diagnostics is the tool you run when things are broken — it must cope."""
    _isolate_mcp_config(monkeypatch, tmp_path)
    agent = _agent()

    monkeypatch.setattr(agent.mcp_manager, "configured_servers", lambda: {})
    assert run_checks(agent)


# ─── reasoning-token extraction ─────────────────────────────


class _Usage:
    def __init__(self, completion=100, reasoning=40, with_details=True):
        self.completion_tokens = completion
        if with_details:
            from types import SimpleNamespace

            self.completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning)


def test_reasoning_tokens_are_read_from_completion_details() -> None:
    assert _openai_reasoning_tokens(_Usage(completion=100, reasoning=40)) == 40


def test_reasoning_tokens_are_clamped_to_completion_tokens() -> None:
    """Reasoning is a subset of output — it can never exceed it."""
    assert _openai_reasoning_tokens(_Usage(completion=10, reasoning=99)) == 10


def test_missing_reasoning_details_read_as_zero() -> None:
    assert _openai_reasoning_tokens(_Usage(with_details=False)) == 0
    assert _openai_reasoning_tokens(object()) == 0


def test_none_reasoning_count_reads_as_zero() -> None:
    assert _openai_reasoning_tokens(_Usage(reasoning=None)) == 0


# ─── agent-side pricing and round accounting ────────────────


def test_switch_model_refreshes_the_rate_card() -> None:
    agent = _agent(model="deepseek-v4.1-flash")
    assert agent.price_resolution.estimated is True

    agent.switch_model("gpt-4o")

    assert agent.price_resolution.match == "exact"
    assert agent.price_resolution.estimated is False


def test_round_tokens_are_a_delta_from_the_chat_snapshot() -> None:
    agent = _agent()
    agent.total_input_tokens = 1000
    agent.total_output_tokens = 20
    agent._round_base = agent._usage_totals()  # what chat() does on entry

    agent.total_input_tokens += 250
    agent.total_output_tokens += 3
    agent.total_reasoning_tokens += 7

    round_tokens = agent._round_tokens()
    assert round_tokens["input"] == 250
    assert round_tokens["output"] == 3
    assert round_tokens["reasoning"] == 7


def test_snapshot_marks_estimated_cost_and_hides_credentials() -> None:
    agent = _agent()

    snapshot = agent.get_status_snapshot()

    assert snapshot["cost_estimated"] is True
    assert "api_key" not in snapshot
    assert "api_base" not in snapshot


def test_clear_history_resets_round_and_reasoning_counters() -> None:
    agent = _agent()
    agent.total_reasoning_tokens = 500
    agent.total_input_tokens = 900

    agent.clear_history()

    assert agent.total_reasoning_tokens == 0
    assert agent._round_tokens()["input"] == 0
