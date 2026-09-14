from pathlib import Path

import pytest

from mini_claude import skills


@pytest.fixture(autouse=True)
def _reset_skill_cache() -> None:
    skills.reset_skill_cache()
    yield
    skills.reset_skill_cache()


def _write_skill(base: Path, name: str, prompt: str) -> None:
    skill_dir = base / ".claude" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description\n---\n\n{prompt}\n",
        encoding="utf-8",
    )


def test_discovers_bundled_skill_outside_install_root(tmp_path, monkeypatch) -> None:
    install_root = tmp_path / "install"
    user_home = tmp_path / "home"
    other_project = tmp_path / "work"
    other_project.mkdir()
    _write_skill(install_root, "phm-training", "bundled prompt")

    monkeypatch.setattr(skills, "INSTALL_ROOT", install_root)
    monkeypatch.setattr(skills.Path, "home", classmethod(lambda cls: user_home))
    monkeypatch.chdir(other_project)

    skill = skills.get_skill_by_name("phm-training")

    assert skill is not None
    assert skill.source == "bundled"
    assert skill.prompt_template.strip() == "bundled prompt"


def test_skill_priority_is_project_then_user_then_bundled(tmp_path, monkeypatch) -> None:
    install_root = tmp_path / "install"
    user_home = tmp_path / "home"
    project_root = tmp_path / "project"
    project_root.mkdir()

    _write_skill(install_root, "shared", "bundled prompt")
    _write_skill(user_home, "shared", "user prompt")
    _write_skill(install_root, "user-wins", "bundled prompt")
    _write_skill(user_home, "user-wins", "user prompt")
    _write_skill(project_root, "shared", "project prompt")

    monkeypatch.setattr(skills, "INSTALL_ROOT", install_root)
    monkeypatch.setattr(skills.Path, "home", classmethod(lambda cls: user_home))
    monkeypatch.chdir(project_root)

    shared = skills.get_skill_by_name("shared")
    user_wins = skills.get_skill_by_name("user-wins")

    assert shared is not None
    assert shared.source == "project"
    assert shared.prompt_template.strip() == "project prompt"
    assert user_wins is not None
    assert user_wins.source == "user"
    assert user_wins.prompt_template.strip() == "user prompt"
