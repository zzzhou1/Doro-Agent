from __future__ import annotations

from prompt_toolkit.utils import get_cwidth

from doro.status import (
    format_status_lines,
    format_status_toolbar,
    format_token_limit,
    truncate_middle,
    truncate_start,
)


def _snapshot(**overrides):
    value = {
        "cwd": r"F:\WorkSpace\demo",
        "context_used": 3072,
        "context_window": 128000,
        "auto_compact": True,
        "cost_usd": 0.0132,
        "session_id": "ab12cd34",
        "backend": "openai",
        "model": "gpt-5.6-sol",
        "permission_mode": "default",
        "thinking_mode": "disabled",
        "reasoning_effort": "medium",
        "processing": False,
    }
    value.update(overrides)
    return value


def _width(text: str) -> int:
    return sum(get_cwidth(char) for char in text)


def test_wide_status_matches_expected_information_hierarchy() -> None:
    divider, path, info = format_status_lines(_snapshot(), width=120)

    assert _width(divider) == 120
    assert path == r"F:\WorkSpace\demo"
    assert "2.4%/128K (auto)" in info
    assert "$0.0132" in info
    assert "ab12cd34" in info
    assert "(openai) gpt-5.6-sol • normal · think:medium" in info
    assert _width(info) == 120


def test_processing_and_plan_modes_are_visible() -> None:
    _divider, _path, working = format_status_lines(
        _snapshot(processing=True), width=100
    )
    assert working.endswith("• working")

    _divider, _path, plan = format_status_lines(
        _snapshot(permission_mode="plan"), width=100
    )
    assert "• plan" in plan
    assert plan.endswith("think:medium")


def test_reasoning_effort_is_visible_when_idle() -> None:
    _divider, _path, info = format_status_lines(
        _snapshot(reasoning_effort="high"), width=110
    )
    assert "normal · think:high" in info


def test_narrow_status_never_exceeds_terminal_width() -> None:
    for width in (20, 32, 48, 60):
        divider, path, info = format_status_lines(
            _snapshot(cwd=r"F:\a-very-long-project-name\nested\folder"), width=width
        )
        assert _width(divider) == width
        assert _width(path) <= width
        assert _width(info) <= width
        assert "2.4%/128K" in info


def test_middle_truncation_respects_wide_characters() -> None:
    result = truncate_middle(r"F:\项目\很长的目录\demo", 12)
    assert _width(result) <= 12
    assert "…" in result
    assert result.endswith("demo")

    suffix = truncate_start("很长的流式输出内容demo", 10)
    assert _width(suffix) <= 10
    assert suffix.startswith("…")
    assert suffix.endswith("demo")


def test_prompt_toolbar_is_single_line_and_keeps_required_fields() -> None:
    line = format_status_toolbar(_snapshot(), width=100)

    assert "\n" not in line
    assert "2.4%/128K auto" in line
    assert "openai:gpt-5.6-sol" in line
    assert _width(line) == 100


def test_prompt_toolbar_auto_width_reserves_the_terminal_final_column(
    monkeypatch,
) -> None:
    monkeypatch.setattr("doro.status.terminal_width", lambda: 100)

    line = format_status_toolbar(_snapshot())

    assert _width(line) == 99
    assert line.endswith("normal · think:medium")


def test_prompt_toolbar_stays_single_line_when_narrow() -> None:
    for width in (20, 32, 48, 60):
        line = format_status_toolbar(_snapshot(), width=width)
        assert "\n" not in line
        assert _width(line) <= width
        assert "2.4%/128K" in line


def test_token_limit_units_are_compact() -> None:
    assert format_token_limit(512) == "512"
    assert format_token_limit(128000) == "128K"
    assert format_token_limit(1_000_000) == "1M"
