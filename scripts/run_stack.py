"""Cross-platform stack launcher for BoilerMind.

Replaces the legacy PowerShell scripts (start-all.ps1, server/start-backend.ps1,
frontend/start-frontend.ps1) with a single Python entry point that:

- works identically on Windows / macOS / Linux;
- picks the project `.venv` Python first, falling back to the historical
  `D:\\anaconda\\envs\\boilermind311\\python.exe` and finally to the system
  `python`;
- loads `.env.local` and the handoff environment variables without printing
  secret values;
- waits for each service to become healthy before declaring success;
- writes PIDs to `runtime/` so `down` and `status` can find the processes later;
- keeps the children alive until SIGINT / SIGTERM, then cleans up.

Usage (from project root):

    python scripts/run_stack.py up                 # backend + frontend + Unity
    python scripts/run_stack.py up --no-unity      # only backend + frontend
    python scripts/run_stack.py backend            # backend only
    python scripts/run_stack.py frontend           # frontend (with Unity) only
    python scripts/run_stack.py frontend --no-unity
    python scripts/run_stack.py down               # stop everything tracked by runtime/*.pid
    python scripts/run_stack.py status             # print who is listening
    python scripts/run_stack.py doctor             # check venv / .env.local / data path
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
VENV_PY_WIN = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PY_UNIX = PROJECT_ROOT / ".venv" / "bin" / "python"
LEGACY_CONDA_PY = Path(r"D:\anaconda\envs\boilermind311\python.exe")
SYSTEM_PY = "python"

DEFAULT_BACKEND_PORT = 8765
DEFAULT_FRONTEND_PORT = 8081
DEFAULT_UNITY_PORT = 8090

# State files written under runtime/. Down/status read them.
BACKEND_PID = RUNTIME_DIR / "backend.pid"
FRONTEND_PID = RUNTIME_DIR / "frontend.pid"
UNITY_PID = RUNTIME_DIR / "unity.pid"
BACKEND_LOG = RUNTIME_DIR / "backend.log"
FRONTEND_LOG = RUNTIME_DIR / "frontend.log"
UNITY_LOG = RUNTIME_DIR / "unity.log"
BACKEND_ERR = RUNTIME_DIR / "backend.err.log"
FRONTEND_ERR = RUNTIME_DIR / "frontend.err.log"
UNITY_ERR = RUNTIME_DIR / "unity.err.log"

HEALTH_TIMEOUT_S = 90
POLL_INTERVAL_S = 0.5


@dataclass
class ServiceSpec:
    name: str
    port: int
    cmd: list[str]
    cwd: Path
    log_out: Path
    log_err: Path
    pid_file: Path
    health_path: str = "/health/ready"
    env_overrides: dict[str, str] = field(default_factory=dict)

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.health_path}"


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a dotenv file into os.environ without echoing."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum() and key[0].isalpha():
            # Do not overwrite anything already set on the shell so callers can
            # override at invocation time (e.g. DASHSCOPE_API_KEY on the cmd line).
            os.environ.setdefault(key, value)


def apply_handoff_env() -> None:
    """Mirror what handoff_env.ps1 used to set."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault(
        "BOILERMIND_QWEN_BASE_URL",
        "https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    os.environ.setdefault("BOILERMIND_QWEN_MODEL", "qwen3.7-plus")
    os.environ.setdefault("BOILERMIND_ENABLE_WEB_LITERATURE", "0")
    os.environ.setdefault(
        "BOILERMIND_REAL_DATASET_PATH",
        str(PROJECT_ROOT / "resources" / "data" / "shortperiod_new.csv"),
    )


def resolve_python() -> str:
    """Pick the best Python: project .venv → legacy conda → system python."""
    candidates = []
    if os.name == "nt":
        if VENV_PY_WIN.is_file():
            candidates.append(str(VENV_PY_WIN))
    else:
        if VENV_PY_UNIX.is_file():
            candidates.append(str(VENV_PY_UNIX))
    if LEGACY_CONDA_PY.is_file():
        candidates.append(str(LEGACY_CONDA_PY))
    candidates.append(SYSTEM_PY)
    for cand in candidates:
        try:
            result = subprocess.run(
                [cand, "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    raise RuntimeError("No usable Python interpreter found.")


# ---------------------------------------------------------------------------
# Port + health checks
# ---------------------------------------------------------------------------


def port_listening(port: int, timeout: float = 0.3) -> bool:
    """Return True if a TCP connection to 127.0.0.1:port succeeds."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_health(spec: ServiceSpec, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
    """Poll the service's health URL until it returns 2xx or we time out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if port_listening(spec.port):
            try:
                with urllib.request.urlopen(spec.health_url(), timeout=2) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
        time.sleep(POLL_INTERVAL_S)
    return False


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def start_service(spec: ServiceSpec, python: str) -> subprocess.Popen:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    # Truncate previous logs so the user sees a clean run.
    for log in (spec.log_out, spec.log_err):
        log.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    for k, v in spec.env_overrides.items():
        env[k] = v
    out = spec.log_out.open("a", encoding="utf-8")
    err = spec.log_err.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        spec.cmd,
        cwd=str(spec.cwd),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
    )
    spec.pid_file.write_text(str(proc.pid), encoding="utf-8")
    return proc


def stop_by_pid_file(pid_file: Path) -> bool:
    if not pid_file.is_file():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return False
    if pid <= 0:
        pid_file.unlink(missing_ok=True)
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    pid_file.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_backend_spec(port: int, python: str) -> ServiceSpec:
    return ServiceSpec(
        name="backend",
        port=port,
        cmd=[
            python,
            "-m",
            "uvicorn",
            "server.research_api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        log_out=BACKEND_LOG,
        log_err=BACKEND_ERR,
        pid_file=BACKEND_PID,
        health_path="/health/ready",
    )


def build_frontend_spec(port: int, unity_port: int, with_unity: bool, python: str) -> list[ServiceSpec]:
    specs: list[ServiceSpec] = []
    primary = ServiceSpec(
        name="frontend",
        port=port,
        cmd=[
            python,
            str(PROJECT_ROOT / "frontend" / "serve_frontend.py"),
            "--port",
            str(port),
            "--directory",
            str(PROJECT_ROOT / "frontend"),
        ],
        cwd=PROJECT_ROOT / "frontend",
        log_out=FRONTEND_LOG,
        log_err=FRONTEND_ERR,
        pid_file=FRONTEND_PID,
        # No dedicated /health endpoint; treat TCP listen as the readiness probe.
        health_path="/",
    )
    specs.append(primary)
    if with_unity:
        unity_root = PROJECT_ROOT / "UnityWebgl"
        unity_spec = ServiceSpec(
            name="unity",
            port=unity_port,
            cmd=[
                python,
                "-m",
                "http.server",
                str(unity_port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(unity_root),
            ],
            cwd=unity_root,
            log_out=UNITY_LOG,
            log_err=UNITY_ERR,
            pid_file=UNITY_PID,
            health_path="/index_unity_only.html",
        )
        specs.append(unity_spec)
    return specs


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_up(args: argparse.Namespace) -> int:
    apply_handoff_env()
    load_dotenv(PROJECT_ROOT / ".env.local")
    python = resolve_python()
    print(f"[run_stack] using python: {python}")
    services: list[ServiceSpec] = [build_backend_spec(args.backend_port, python)]
    services += build_frontend_spec(
        args.frontend_port, args.unity_port, not args.no_unity, python
    )
    # If something is already up, leave it alone and reuse it (idempotent).
    pending = []
    for spec in services:
        if port_listening(spec.port):
            print(f"[run_stack] {spec.name} :{spec.port} already running")
            continue
        pending.append(spec)
    if not pending:
        print("[run_stack] everything is already up")
        return 0
    procs: list[subprocess.Popen] = []
    for spec in pending:
        print(f"[run_stack] starting {spec.name} :{spec.port} ...")
        procs.append(start_service(spec, python))
    failed: list[ServiceSpec] = []
    for spec in pending:
        if wait_for_health(spec, HEALTH_TIMEOUT_S):
            print(f"[run_stack] {spec.name} :{spec.port} ready")
        else:
            print(f"[run_stack] {spec.name} :{spec.port} FAILED (timeout, see {spec.log_err})")
            failed.append(spec)
    if failed:
        for spec in failed:
            stop_by_pid_file(spec.pid_file)
        return 1
    _print_summary(args)
    if args.detach:
        return 0
    _wait_for_signal(procs)
    return 0


def _wait_for_signal(procs: list[subprocess.Popen]) -> None:
    print("[run_stack] all services running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            for p in procs:
                if p.poll() is not None:
                    print(f"[run_stack] child exited with code {p.returncode}; aborting")
                    return
    except KeyboardInterrupt:
        print("\n[run_stack] received SIGINT, shutting down...")
    finally:
        cmd_down(argparse.Namespace())


def cmd_backend(args: argparse.Namespace) -> int:
    apply_handoff_env()
    load_dotenv(PROJECT_ROOT / ".env.local")
    python = resolve_python()
    spec = build_backend_spec(args.backend_port, python)
    if port_listening(spec.port):
        print(f"[run_stack] backend :{spec.port} already running")
        return 0
    print(f"[run_stack] starting backend :{spec.port} ...")
    start_service(spec, python)
    if wait_for_health(spec, HEALTH_TIMEOUT_S):
        print(f"[run_stack] backend :{spec.port} ready")
        return 0
    print(f"[run_stack] backend FAILED (see {spec.log_err})")
    stop_by_pid_file(spec.pid_file)
    return 1


def cmd_frontend(args: argparse.Namespace) -> int:
    apply_handoff_env()
    python = resolve_python()
    services = build_frontend_spec(
        args.frontend_port, args.unity_port, not args.no_unity, python
    )
    procs: list[subprocess.Popen] = []
    for spec in services:
        if port_listening(spec.port):
            print(f"[run_stack] {spec.name} :{spec.port} already running")
            continue
        print(f"[run_stack] starting {spec.name} :{spec.port} ...")
        procs.append(start_service(spec, python))
    for spec in services:
        if port_listening(spec.port):
            print(f"[run_stack] {spec.name} :{spec.port} ready")
        else:
            print(f"[run_stack] {spec.name} :{spec.port} FAILED")
    if procs and not args.detach:
        _wait_for_signal(procs)
    return 0


def cmd_down(_: argparse.Namespace) -> int:
    for pid_file in (BACKEND_PID, FRONTEND_PID, UNITY_PID):
        if stop_by_pid_file(pid_file):
            print(f"[run_stack] stopped pid from {pid_file.name}")
    # Sweep anything still listening on the default ports in case the
    # PID files were lost (e.g. crashed shell).
    for port in (DEFAULT_BACKEND_PORT, DEFAULT_FRONTEND_PORT, DEFAULT_UNITY_PORT):
        _kill_listener_on(port)
    return 0


def _kill_listener_on(port: int) -> None:
    if not port_listening(port):
        return
    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction SilentlyContinue | "
                    f"ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}",
                ],
                capture_output=True,
                check=False,
            )
        else:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                check=False,
            )
    except OSError:
        pass


def cmd_status(_: argparse.Namespace) -> int:
    rows = []
    for label, port, pid_file in (
        ("backend", DEFAULT_BACKEND_PORT, BACKEND_PID),
        ("frontend", DEFAULT_FRONTEND_PORT, FRONTEND_PID),
        ("unity", DEFAULT_UNITY_PORT, UNITY_PID),
    ):
        listening = port_listening(port)
        pid = "n/a"
        if pid_file.is_file():
            pid = pid_file.read_text(encoding="utf-8").strip() or "n/a"
        rows.append((label, port, "UP" if listening else "DOWN", pid))
    width = max(len(r[0]) for r in rows)
    for label, port, state, pid in rows:
        print(f"  {label.ljust(width)} :{port:<5} {state:<5} pid={pid}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    apply_handoff_env()
    issues: list[str] = []
    py = None
    try:
        py = resolve_python()
        print(f"[doctor] python: {py}")
    except RuntimeError as exc:
        issues.append(str(exc))
        print(f"[doctor] python: MISSING ({exc})")
    if py is not None:
        try:
            result = subprocess.run(
                [py, "-c", "import boilermind"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                check=False,
            )
            if result.returncode != 0:
                issues.append("boilermind not importable")
                print(f"[doctor] boilermind: MISSING ({result.stderr.strip().splitlines()[-1] if result.stderr else '?'})")
            else:
                print("[doctor] boilermind: importable")
        except OSError as exc:
            issues.append(str(exc))
            print(f"[doctor] boilermind: ERROR ({exc})")
    dotenv = PROJECT_ROOT / ".env.local"
    if dotenv.is_file():
        print("[doctor] .env.local: present")
    else:
        # Optional: doctor still passes if .env.local is absent because the
        # legacy handoff env + system defaults are enough to run the backend.
        print("[doctor] .env.local: absent (OK if running with defaults)")
    dataset = Path(os.environ.get("BOILERMIND_REAL_DATASET_PATH", ""))
    if not str(dataset) or str(dataset) == ".":
        dataset = PROJECT_ROOT / "resources" / "data" / "shortperiod_new.csv"
    if dataset.is_file():
        print(f"[doctor] dataset: {dataset}")
    else:
        issues.append(f"dataset missing: {dataset}")
        print(f"[doctor] dataset: MISSING ({dataset})")
    print("[doctor] OK" if not issues else f"[doctor] issues: {issues}")
    return 0 if not issues else 1


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_stack",
        description="Cross-platform launcher for the BoilerMind research stack.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    common.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    common.add_argument("--unity-port", type=int, default=DEFAULT_UNITY_PORT)
    common.add_argument("--no-unity", action="store_true", help="skip Unity web server")
    common.add_argument(
        "--detach",
        action="store_true",
        help="start children and return immediately (do not wait for SIGINT)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up", parents=[common], help="start backend + frontend (default: with Unity)")
    sub.add_parser("backend", parents=[common], help="start the FastAPI backend only")
    sub.add_parser("frontend", parents=[common], help="start the static frontend (+Unity)")
    sub.add_parser("down", help="stop everything tracked by runtime/*.pid")
    sub.add_parser("status", help="show which ports are listening")
    sub.add_parser("doctor", help="validate python/env/dataset")
    return parser


def _print_summary(args: argparse.Namespace) -> None:
    print()
    print("[run_stack] ready.")
    print(f"  backend  : http://127.0.0.1:{args.backend_port}/health/ready")
    print(f"  frontend : http://127.0.0.1:{args.frontend_port}/")
    if not args.no_unity:
        print(f"  unity    : http://127.0.0.1:{args.unity_port}/index_unity_only.html")
    print(f"  logs     : {RUNTIME_DIR}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handlers = {
        "up": cmd_up,
        "backend": cmd_backend,
        "frontend": cmd_frontend,
        "down": cmd_down,
        "status": cmd_status,
        "doctor": cmd_doctor,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
