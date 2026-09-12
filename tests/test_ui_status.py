from __future__ import annotations

from types import SimpleNamespace

import mini_claude.ui as ui


def _snapshot():
    return {
        "cwd": r"F:\WorkSpace",
        "context_used": 0,
        "context_window": 128000,
        "backend": "openai",
        "model": "gpt-test",
    }


def test_live_renderable_uses_shared_status_layout() -> None:
    renderable = ui._LiveStatusRenderable(_snapshot)
    rendered = list(
        renderable.__rich_console__(None, SimpleNamespace(max_width=80))
    )

    assert len(rendered) == 1
    assert r"F:\WorkSpace" in rendered[0].plain
    assert "0.0%/128K (auto)" in rendered[0].plain
    assert "gpt-test" in rendered[0].plain


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


def test_live_status_is_disabled_for_redirected_output(monkeypatch) -> None:
    ui.stop_live_status()
    monkeypatch.setattr(ui, "console", SimpleNamespace(is_terminal=False))

    assert ui.start_live_status(_snapshot) is False
    assert ui.live_status_active() is False
