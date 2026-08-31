from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if process.poll() is None:
            process.kill()
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", default="outputs/teammate/model_runs")
    parser.add_argument("--status-dir", default="outputs/teammate/supervisor")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    task_file = Path(args.task_file).resolve()
    bundle = json.loads(task_file.read_text(encoding="utf-8"))
    task = next((item for item in bundle.get("tasks", []) if item.get("task_id") == args.task_id), None)
    if task is None or task.get("execution_authorization") != "SMOKE_APPROVED":
        raise SystemExit(f"task_missing_or_not_authorized:{args.task_id}")
    command = [sys.executable, str(root / "scripts" / "run_teammate_task.py"),
               "--task-file", str(task_file), "--task-id", args.task_id, "--output-dir", args.output_dir]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    extra = {"start_new_session": True} if os.name != "nt" else {}
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace", creationflags=flags, **extra)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=float(task["timeout_seconds"]))
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop(process)
        stdout, stderr = process.communicate()
    payload = {"schema_version": "boilermind.teammate_supervisor.v1", "task_id": args.task_id,
               "status": "ENGINEERING_TIMEOUT" if timed_out else ("COMPLETED" if process.returncode == 0 else "FAILED"),
               "return_code": process.returncode, "runtime_seconds": time.perf_counter() - started,
               "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:],
               "execution_artifacts_eligible_for_scientific_audit": not timed_out and process.returncode == 0,
               "scientific_conclusion_established": False,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    status_dir = (root / args.status_dir).resolve()
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{args.task_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 124 if timed_out else int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
