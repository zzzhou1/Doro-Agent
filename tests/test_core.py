from __future__ import annotations

from pathlib import Path

from mini_claude.frontmatter import format_frontmatter, parse_frontmatter
from mini_claude import session


def test_frontmatter_round_trip() -> None:
    rendered = format_frontmatter(
        {"name": "reviewer", "allowed-tools": "read_file,grep_search"},
        "Review the requested files.",
    )
    parsed = parse_frontmatter(rendered)
    assert parsed.meta == {
        "name": "reviewer",
        "allowed-tools": "read_file,grep_search",
    }
    assert parsed.body == "Review the requested files."


def test_session_round_trip(tmp_path: Path, monkeypatch) -> None:
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(session, "SESSION_DIR", session_dir)
    payload = {
        "metadata": {"id": "test-session", "startTime": "2026-01-01T00:00:00Z"},
        "anthropicMessages": [{"role": "user", "content": "hello"}],
    }

    session.save_session("test-session", payload)

    assert session.load_session("test-session") == payload
    assert session.get_latest_session_id() == "test-session"
