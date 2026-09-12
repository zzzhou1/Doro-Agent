"""Session management — JSON file persistence for conversation history."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

SESSION_DIR = Path.home() / ".mini-claude" / "sessions"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _ensure_dir() -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("Invalid session ID")
    return SESSION_DIR / f"{session_id}.json"


def save_session(session_id: str, data: dict[str, Any]) -> None:
    path = _session_path(session_id)
    _ensure_dir()
    temp_path = SESSION_DIR / f".{session_id}.{uuid.uuid4().hex}.tmp"
    try:
        temp_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_session(session_id: str) -> dict[str, Any] | None:
    try:
        path = _session_path(session_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalized_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _conversation_preview(data: dict[str, Any], backend: str | None) -> str:
    key = "openaiMessages" if backend == "openai" else "anthropicMessages"
    messages = data.get(key)
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = content.split("\n\n<system-reminder>", 1)[0]
        preview = " ".join(content.split())[:120]
        if preview:
            return preview
    return ""


def list_sessions(
    *,
    cwd: str | Path | None = None,
    backend: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not SESSION_DIR.exists():
        return []
    expected_cwd = _normalized_path(cwd) if cwd is not None else None
    results: list[dict[str, Any]] = []
    for f in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("id") != f.stem or not SESSION_ID_PATTERN.fullmatch(f.stem):
                continue
            if expected_cwd is not None:
                stored_cwd = metadata.get("cwd")
                if not stored_cwd or _normalized_path(stored_cwd) != expected_cwd:
                    continue
            stored_backend = metadata.get("backend")
            if stored_backend is None:
                if data.get("openaiMessages") is not None:
                    stored_backend = "openai"
                elif data.get("anthropicMessages") is not None:
                    stored_backend = "anthropic"
            if backend is not None and stored_backend != backend:
                continue
            if not metadata.get("preview"):
                metadata = dict(metadata)
                metadata["preview"] = _conversation_preview(data, stored_backend)
            results.append(metadata)
        except Exception:
            pass
    results.sort(
        key=lambda item: item.get("updatedAt") or item.get("startTime", ""),
        reverse=True,
    )
    return results[:limit] if limit is not None else results


def get_latest_session_id(
    *, cwd: str | Path | None = None, backend: str | None = None
) -> str | None:
    sessions = list_sessions(cwd=cwd, backend=backend, limit=1)
    if not sessions:
        return None
    return sessions[0].get("id")
