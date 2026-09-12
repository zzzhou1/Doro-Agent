from __future__ import annotations

from pathlib import Path

import pytest

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
    assert list(session_dir.glob("*.tmp")) == []


def test_sessions_are_filtered_and_sorted(tmp_path: Path, monkeypatch) -> None:
    session_dir = tmp_path / "sessions"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    monkeypatch.setattr(session, "SESSION_DIR", session_dir)

    payloads = [
        ("older", project_a, "openai", "2026-01-01T00:00:00Z"),
        ("newer", project_a, "openai", "2026-01-03T00:00:00Z"),
        ("anthropic", project_a, "anthropic", "2026-01-04T00:00:00Z"),
        ("other-project", project_b, "openai", "2026-01-05T00:00:00Z"),
    ]
    for session_id, cwd, backend, updated_at in payloads:
        session.save_session(session_id, {
            "metadata": {
                "id": session_id,
                "cwd": str(cwd),
                "backend": backend,
                "startTime": "2026-01-01T00:00:00Z",
                "updatedAt": updated_at,
            },
            "openaiMessages": [] if backend == "openai" else None,
            "anthropicMessages": [] if backend == "anthropic" else None,
        })

    matches = session.list_sessions(cwd=project_a, backend="openai")
    assert [item["id"] for item in matches] == ["newer", "older"]
    assert session.get_latest_session_id(cwd=project_a, backend="openai") == "newer"


def test_legacy_session_backend_is_inferred(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")
    session.save_session("legacy", {
        "metadata": {"id": "legacy", "cwd": str(tmp_path)},
        "openaiMessages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "legacy question\n\n<system-reminder>hidden memory</system-reminder>",
            },
        ],
    })

    matches = session.list_sessions(cwd=tmp_path, backend="openai")
    assert [item["id"] for item in matches] == ["legacy"]
    assert matches[0]["preview"] == "legacy question"


def test_invalid_session_id_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "sessions")

    with pytest.raises(ValueError, match="Invalid session ID"):
        session.save_session("../escape", {"metadata": {}})
    assert session.load_session("../escape") is None
