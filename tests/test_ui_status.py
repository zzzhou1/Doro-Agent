from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mini_claude.ui as ui


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


def test_live_status_is_disabled_for_redirected_output(monkeypatch) -> None:
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", SimpleNamespace(is_terminal=False))

    assert ui.start_live_status(_snapshot) is False
    assert ui.live_status_active() is False
