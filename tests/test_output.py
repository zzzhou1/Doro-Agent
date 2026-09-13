"""Tests for the unified output coordinator (mini_claude.output)."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import mini_claude.ui as ui
from mini_claude.output import INFO, WARNING, OutputCoordinator, OutputEvent


def test_emit_dispatches_to_registered_renderer_in_order() -> None:
    coordinator = OutputCoordinator()
    seen: list[OutputEvent] = []
    coordinator.set_renderer(seen.append)

    coordinator.emit(OutputEvent(INFO, text="a"))
    coordinator.emit(OutputEvent(WARNING, text="b"))

    assert [event.kind for event in seen] == [INFO, WARNING]
    assert [event.text for event in seen] == ["a", "b"]


def test_renderer_failure_falls_back_without_raising(capsys) -> None:
    """Output is diagnostic — a broken renderer must not kill the caller."""
    coordinator = OutputCoordinator()

    def broken(_event: OutputEvent) -> None:
        raise RuntimeError("boom")

    coordinator.set_renderer(broken)
    coordinator.emit(OutputEvent(INFO, text="still shown"))

    assert "still shown" in capsys.readouterr().out


def test_emit_is_serialized_across_threads() -> None:
    coordinator = OutputCoordinator()
    seen: list[str] = []
    coordinator.set_renderer(lambda event: seen.append(event.text))

    def worker(n: int) -> None:
        for i in range(50):
            coordinator.emit(OutputEvent(INFO, text=f"{n}-{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 200
    assert len(set(seen)) == 200


def test_ui_warning_preserves_bracket_text(monkeypatch) -> None:
    """[mcp]-style prefixes must render literally, not as Rich style tags."""
    ui.stop_live_status()
    fake_console = Mock()
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_warning("[mcp] Connected to 'amap' [read-only]")

    call = fake_console.print.call_args
    assert "[mcp] Connected to 'amap' [read-only]" in call.args[0]
    assert call.kwargs["markup"] is False


def test_tool_result_preserves_bracket_text(monkeypatch) -> None:
    """Regression: tool output containing [..] was silently eaten by markup."""
    ui.stop_live_status()
    fake_console = Mock()
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_tool_result("run_shell", "echo [mcp] done")

    call = fake_console.print.call_args
    assert "[mcp]" in call.args[0]
    assert call.kwargs["markup"] is False


def test_file_change_result_preserves_bracket_text(monkeypatch) -> None:
    ui.stop_live_status()
    fake_console = Mock()
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_tool_result("edit_file", "file.py\n+ x = [1, 2]")

    rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list)
    assert "+ x = [1, 2]" in rendered


def test_public_api_routes_every_kind_through_the_renderer() -> None:
    """Every ui.print_* wrapper must emit an event the registry handles."""
    from mini_claude import output

    for kind in (
        output.ASSISTANT_DELTA,
        output.TOOL_START,
        output.TOOL_RESULT,
        output.WARNING,
        output.INFO,
        output.ERROR,
        output.CONFIRMATION,
        output.RETRY,
        output.USAGE,
        output.PANEL,
        output.DIVIDER,
        output.WELCOME,
        output.USER_PROMPT,
        output.PLAN_APPROVAL,
        output.PLAN_OPTIONS,
        output.SUBAGENT_START,
        output.SUBAGENT_END,
    ):
        assert kind in ui._RENDERERS, kind
