from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import doro.ui as ui


def _snapshot():
    return {
        "cwd": r"F:\WorkSpace",
        "context_used": 0,
        "context_window": 128000,
        "backend": "openai",
        "model": "gpt-test",
    }


def test_live_renderable_is_single_line_and_avoids_the_final_column() -> None:
    renderable = ui._LiveStatusRenderable(_snapshot)
    rendered = list(
        renderable.__rich_console__(None, SimpleNamespace(max_width=80))
    )

    assert len(rendered) == 1
    line = rendered[0].plain
    assert r"F:\WorkSpace" in line
    assert "0.0%/128K auto" in line
    assert "gpt-test" in line
    assert "\n" not in line
    assert "─" not in line
    assert len(line) == 79


def test_live_status_lifecycle_and_suspend(monkeypatch) -> None:
    instances = []

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            self.started = False
            self.stopped = False
            self.refresh_count = 0
            instances.append(self)

        def start(self, refresh=False):
            self.started = refresh

        def refresh(self):
            self.refresh_count += 1

        def stop(self):
            self.stopped = True

    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", SimpleNamespace(is_terminal=True))
    monkeypatch.setattr(ui, "Live", FakeLive)

    assert ui.start_live_status(_snapshot) is True
    assert ui.live_status_active() is True
    ui.refresh_live_status()
    assert instances[0].refresh_count == 1

    with ui.suspend_live_status():
        assert ui.live_status_active() is False
        assert instances[0].stopped is True

    assert ui.live_status_active() is True
    assert len(instances) == 2
    ui.stop_live_status()
    assert instances[1].stopped is True


def test_streaming_chunks_are_buffered_until_a_complete_line(monkeypatch) -> None:
    instances = []

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            self.refresh_count = 0
            self.stopped = False
            instances.append(self)

        def start(self, refresh=False):
            pass

        def refresh(self):
            self.refresh_count += 1

        def stop(self):
            self.stopped = True

    fake_console = SimpleNamespace(is_terminal=True, print=Mock())
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", fake_console)
    monkeypatch.setattr(ui, "Live", FakeLive)

    assert ui.start_live_status(_snapshot) is True
    ui.print_assistant_text("收到")
    assert fake_console.print.call_count == 0
    assert instances[0].refresh_count == 1

    ui.print_assistant_text("正常\n下一行")
    assert fake_console.print.call_count == 1
    assert fake_console.print.call_args.args == ("收到正常\n",)
    assert fake_console.print.call_args.kwargs["end"] == ""

    ui.stop_live_status()
    assert instances[0].stopped is True
    assert fake_console.print.call_count == 2
    assert fake_console.print.call_args.args == ("下一行",)
    assert fake_console.print.call_args.kwargs["end"] == ""


def test_cost_is_printed_after_the_pending_assistant_line(monkeypatch) -> None:
    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, refresh=False):
            pass

        def refresh(self):
            pass

        def stop(self):
            pass

    fake_console = SimpleNamespace(is_terminal=True, print=Mock())
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", fake_console)
    monkeypatch.setattr(ui, "Live", FakeLive)

    assert ui.start_live_status(_snapshot) is True
    ui.print_assistant_text("第一段\n最后一段")
    ui.print_cost(27469, 98, cache_read_tokens=27469)

    calls = fake_console.print.call_args_list
    assert calls[0].args == ("第一段\n",)
    assert calls[1].args == ("最后一段\n",)
    assert "Tokens: 27469 in (27469 cached) / 98 out" in calls[2].args[0]
    ui.stop_live_status()


def test_end_of_turn_footer_shows_only_this_round(monkeypatch) -> None:
    """One line, this round only: no session row, no "this round:", no "estimated"."""
    fake_console = SimpleNamespace(is_terminal=True, print=Mock())
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_cost(
        10000,
        40,
        cache_read_tokens=9000,
        reasoning_tokens=12,
        session_cost=0.05,
        round_usage={
            "input": 10000,
            "output": 5,
            "cache_read": 9000,
            "cache_write": 0,
            "reasoning": 3,
            "cost": 0.005,
            "calls": 2,
        },
        estimated=True,
    )

    rendered = fake_console.print.call_args.args[0]
    assert "Tokens: 10000 in (90% cached) / 5 out · 3 reasoning · 2 calls (~$0.0050)" in rendered
    # Session totals and the estimate qualifier live in /cost, not in the footer.
    assert "this round" not in rendered
    assert "estimated" not in rendered
    assert "~$0.0500" not in rendered
    assert fake_console.print.call_count == 1


def test_divider_and_footer_do_not_add_blank_lines(monkeypatch) -> None:
    """The rule hugs the answer and the footer hugs the rule."""
    fake_console = SimpleNamespace(is_terminal=True, print=Mock())
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", fake_console)
    # Simulate a streamed answer whose final line never got a newline.
    monkeypatch.setattr(ui, "_line_open", True)

    ui.print_divider()
    ui.print_cost(
        0,
        0,
        round_usage={"input": 100, "output": 5, "cache_read": 90, "cost": 0.001, "calls": 1},
    )

    divider = fake_console.print.call_args_list[0].args[0]
    footer = fake_console.print.call_args_list[1].args[0]
    # Terminates the open line, but never inserts a blank one.
    assert divider.startswith("\n[dim]")
    assert not divider.startswith("\n\n")
    # And the footer sits directly under the rule.
    assert footer.startswith("  Tokens:"), footer


def test_cost_line_without_pricing_falls_back_to_the_discount_rate(monkeypatch) -> None:
    """The legacy call shape (no session_cost) must keep working."""
    fake_console = SimpleNamespace(is_terminal=True, print=Mock())
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_cost(1_000_000, 0, cache_read_tokens=1_000_000, cache_read_discount=0.5)

    assert "~$1.5000" in fake_console.print.call_args.args[0]


def test_panel_renders_a_title_and_unescaped_body(monkeypatch) -> None:
    """Report bodies carry model ids, paths and error text — brackets included."""
    fake_console = SimpleNamespace(is_terminal=True, print=Mock())
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_panel("MCP servers", ["[mcp] error: name must not contain '__'"])

    rendered = "\n".join(
        str(call.args[0]) for call in fake_console.print.call_args_list
    )
    assert "MCP servers" in rendered
    assert "[mcp] error: name must not contain '__'" in rendered
    assert all(
        call.kwargs.get("markup", True) is False
        for call in fake_console.print.call_args_list[1:]
    )


def test_welcome_lists_effort_command(monkeypatch) -> None:
    fake_console = SimpleNamespace(print=Mock())
    monkeypatch.setattr(ui, "console", fake_console)

    ui.print_welcome()

    rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list)
    assert "Doro" in rendered
    assert "Coding + PHM" in rendered
    assert "Commands:" in rendered
    assert "/model /effort /session" in rendered


def test_live_status_is_disabled_for_redirected_output(monkeypatch) -> None:
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", SimpleNamespace(is_terminal=False))

    assert ui.start_live_status(_snapshot) is False
    assert ui.live_status_active() is False
