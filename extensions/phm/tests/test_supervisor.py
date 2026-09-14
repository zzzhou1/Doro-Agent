from pathlib import Path
from types import SimpleNamespace

import phm_mcp.supervisor as supervisor_module
from phm_mcp.supervisor import WorkerSupervisor


def test_ensure_worker_is_detached_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fake_popen)
    supervisor = WorkerSupervisor(tmp_path / "data", tmp_path / "artifacts", tmp_path / "runtime")

    started = supervisor.ensure_worker(idle_timeout=60)
    reused = supervisor.ensure_worker(idle_timeout=60)

    assert started["action"] == "started"
    assert started["alive"] is True
    assert started["pid"] == 4321
    assert reused["action"] == "already_running"
    assert len(calls) == 1
    command, options = calls[0]
    assert command[1:3] == ["-m", "phm_mcp.worker"]
    assert "--idle-timeout" in command
    assert options["stdin"] is supervisor_module.subprocess.DEVNULL
    assert Path(started["stdout_log"]).parent == tmp_path / "runtime"
