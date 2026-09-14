"""Session persistence — a corrupt file must be reported, not silently dropped.

Both ``load_session`` and ``list_sessions`` used to swallow every exception, so
a damaged session file looked exactly like a session that never existed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import doro.session as session


@pytest.fixture()
def session_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    return tmp_path


def test_a_corrupt_session_file_is_reported(session_dir: Path) -> None:
    (session_dir / "broken.json").write_text("{not json", encoding="utf-8")

    with patch.object(session, "emit_warning") as warned:
        assert session.load_session("broken") is None

    assert warned.call_count == 1
    message = warned.call_args[0][0]
    assert "broken" in message
    assert "JSONDecodeError" in message


def test_a_valid_session_file_is_loaded_quietly(session_dir: Path) -> None:
    payload = {"metadata": {"id": "ok"}}
    (session_dir / "ok.json").write_text(json.dumps(payload), encoding="utf-8")

    with patch.object(session, "emit_warning") as warned:
        assert session.load_session("ok") == payload

    warned.assert_not_called()


def test_a_missing_session_file_is_not_an_error(session_dir: Path) -> None:
    with patch.object(session, "emit_warning") as warned:
        assert session.load_session("absent") is None

    warned.assert_not_called()


def test_an_invalid_session_id_is_rejected_quietly(session_dir: Path) -> None:
    with patch.object(session, "emit_warning") as warned:
        assert session.load_session("../escape") is None

    warned.assert_not_called()


def test_listing_skips_corrupt_files_but_keeps_the_rest(session_dir: Path) -> None:
    (session_dir / "broken.json").write_text("{{{", encoding="utf-8")
    (session_dir / "good.json").write_text(
        json.dumps({"metadata": {"id": "good", "updatedAt": "2026-09-13T10:00:00"}}),
        encoding="utf-8",
    )

    with patch.object(session, "emit_warning") as warned:
        listed = session.list_sessions()

    assert [item["id"] for item in listed] == ["good"]
    assert warned.call_count == 1
    assert "broken.json" in warned.call_args[0][0]
