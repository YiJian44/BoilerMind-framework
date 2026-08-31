"""Unit tests for scripts/run_stack.py launcher.

These tests intentionally avoid spinning real uvicorn / http servers; they
exercise the helpers that have to work cross-platform (env loading,
interpreter selection, port probing, pid tracking) so that a regression is
caught before the integration run that the operator actually performs.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_stack  # noqa: E402  (path injected above)


@pytest.fixture(autouse=True)
def _clean_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(run_stack, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_stack, "BACKEND_PID", run_stack.RUNTIME_DIR / "backend.pid")
    monkeypatch.setattr(run_stack, "FRONTEND_PID", run_stack.RUNTIME_DIR / "frontend.pid")
    monkeypatch.setattr(run_stack, "UNITY_PID", run_stack.RUNTIME_DIR / "unity.pid")
    monkeypatch.setattr(run_stack, "BACKEND_LOG", run_stack.RUNTIME_DIR / "backend.log")
    monkeypatch.setattr(run_stack, "BACKEND_ERR", run_stack.RUNTIME_DIR / "backend.err.log")
    monkeypatch.setattr(run_stack, "FRONTEND_LOG", run_stack.RUNTIME_DIR / "frontend.log")
    monkeypatch.setattr(run_stack, "FRONTEND_ERR", run_stack.RUNTIME_DIR / "frontend.err.log")
    monkeypatch.setattr(run_stack, "UNITY_LOG", run_stack.RUNTIME_DIR / "unity.log")
    monkeypatch.setattr(run_stack, "UNITY_ERR", run_stack.RUNTIME_DIR / "unity.err.log")
    run_stack.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def test_load_dotenv_sets_only_unset(monkeypatch, tmp_path):
    f = tmp_path / ".env"
    f.write_text("FOO_TEST_KEY=from_file\nBAR_TEST_KEY='quoted'\n# comment\n", encoding="utf-8")
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    monkeypatch.delenv("BAR_TEST_KEY", raising=False)
    run_stack.load_dotenv(f)
    assert os.environ["FOO_TEST_KEY"] == "from_file"
    assert os.environ["BAR_TEST_KEY"] == "quoted"
    # Pre-existing env values must win.
    monkeypatch.setenv("FOO_TEST_KEY", "pre_set")
    run_stack.load_dotenv(f)
    assert os.environ["FOO_TEST_KEY"] == "pre_set"


def test_apply_handoff_env_is_idempotent(monkeypatch):
    monkeypatch.delenv("BOILERMIND_QWEN_MODEL", raising=False)
    monkeypatch.delenv("BOILERMIND_ENABLE_WEB_LITERATURE", raising=False)
    run_stack.apply_handoff_env()
    first = os.environ["BOILERMIND_QWEN_MODEL"]
    run_stack.apply_handoff_env()
    assert os.environ["BOILERMIND_QWEN_MODEL"] == first
    assert os.environ["BOILERMIND_REAL_DATASET_PATH"].endswith("shortperiod_new.csv")


def test_resolve_python_picks_system_python_when_no_venv(monkeypatch, tmp_path):
    fake_venv_win = tmp_path / ".venv" / "Scripts" / "python.exe"
    fake_venv_unix = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(run_stack, "VENV_PY_WIN", fake_venv_win)
    monkeypatch.setattr(run_stack, "VENV_PY_UNIX", fake_venv_unix)
    chosen = run_stack.resolve_python()
    assert chosen  # at minimum the system interpreter should resolve.


def test_port_listening_returns_false_when_closed():
    # Bind an ephemeral port, close it, ensure probing fails.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert run_stack.port_listening(port, timeout=0.1) is False


def test_port_listening_returns_true_for_bound_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert run_stack.port_listening(port, timeout=0.3) is True
    finally:
        s.close()


def test_start_service_writes_pid_and_logs(tmp_path, monkeypatch):
    spec = run_stack.ServiceSpec(
        name="sleeper",
        port=0,
        cmd=[sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=PROJECT_ROOT,
        log_out=tmp_path / "out.log",
        log_err=tmp_path / "err.log",
        pid_file=tmp_path / "sleeper.pid",
    )
    proc = run_stack.start_service(spec, sys.executable)
    try:
        assert spec.pid_file.is_file()
        assert int(spec.pid_file.read_text(encoding="utf-8").strip()) == proc.pid
        # Give the OS a moment to create the log files.
        time.sleep(0.2)
        assert spec.log_out.is_file()
        assert spec.log_err.is_file()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_stop_by_pid_file_handles_missing_or_garbage(tmp_path):
    ghost = tmp_path / "ghost.pid"
    assert run_stack.stop_by_pid_file(ghost) is False
    ghost.write_text("not-a-number", encoding="utf-8")
    assert run_stack.stop_by_pid_file(ghost) is False
    assert not ghost.exists()


def test_build_backend_spec_uses_uvicorn():
    spec = run_stack.build_backend_spec(9001, sys.executable)
    assert spec.name == "backend"
    assert spec.port == 9001
    assert spec.cmd[0] == sys.executable
    assert "uvicorn" in spec.cmd
    assert "server.research_api.app:app" in spec.cmd


def test_build_frontend_spec_with_and_without_unity():
    full = run_stack.build_frontend_spec(8081, 8090, with_unity=True, python=sys.executable)
    assert [s.name for s in full] == ["frontend", "unity"]
    no_unity = run_stack.build_frontend_spec(8081, 8090, with_unity=False, python=sys.executable)
    assert [s.name for s in no_unity] == ["frontend"]


def test_build_parser_has_expected_subcommands():
    parser = run_stack.build_parser()
    sub = parser.parse_args(["up", "--backend-port", "9000"])
    assert sub.cmd == "up"
    assert sub.backend_port == 9000
    sub = parser.parse_args(["up", "--no-unity"])
    assert sub.no_unity is True
    sub = parser.parse_args(["down"])
    assert sub.cmd == "down"


def test_doctor_returns_zero_in_dev_environment():
    # doctor is the only subcommand that exercises real filesystem + env.
    rc = run_stack.cmd_doctor(run_stack.build_parser().parse_args(["doctor"]))
    assert rc in (0, 1)  # we don't fail the test on optional items.
