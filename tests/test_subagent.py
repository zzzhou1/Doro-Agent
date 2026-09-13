"""Custom sub-agent discovery.

``_load_agents_from_dir`` used to swallow every exception, so a broken
definition was indistinguishable from a missing one: the agent was simply not
offered, with nothing to explain why.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mini_claude.subagent import _load_agents_from_dir


def test_a_broken_definition_is_reported(tmp_path: Path) -> None:
    # A directory named *.md passes the suffix check and then fails to read.
    (tmp_path / "broken.md").mkdir()
    agents: dict[str, dict] = {}

    with patch("mini_claude.subagent.emit_warning") as warned:
        _load_agents_from_dir(tmp_path, agents)

    assert agents == {}
    assert warned.call_count == 1
    assert "broken.md" in warned.call_args[0][0]


def test_a_valid_definition_loads_quietly(tmp_path: Path) -> None:
    (tmp_path / "helper.md").write_text(
        "---\nname: helper\ndescription: does things\n---\nYou help.\n",
        encoding="utf-8",
    )
    agents: dict[str, dict] = {}

    with patch("mini_claude.subagent.emit_warning") as warned:
        _load_agents_from_dir(tmp_path, agents)

    warned.assert_not_called()
    assert agents["helper"]["description"] == "does things"
    assert agents["helper"]["system_prompt"] == "You help."


def test_a_non_markdown_entry_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    agents: dict[str, dict] = {}

    _load_agents_from_dir(tmp_path, agents)

    assert agents == {}
