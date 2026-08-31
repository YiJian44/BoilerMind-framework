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


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if process.poll() is None:
            process.kill()
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description="在独立进程中运行单模型实验并执行硬超时。")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--output-dir", default="outputs/first_batch/model_runs")
    parser.add_argument("--status-dir", default="outputs/experiment_supervisor")
    args = parser.parse_args()
    command = [
        sys.executable, str(Path(__file__).with_name("run_first_batch_model.py")),
        "--model", args.model, "--seed", str(args.seed),
        "--max-epochs", str(args.max_epochs), "--output-dir", args.output_dir,
    ]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    popen_kwargs = {"start_new_session": True} if os.name != "nt" else {}
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace",
                               creationflags=creationflags, **popen_kwargs)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_tree(process)
        stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    status = {
        "schema_version": "boilermind.process_supervisor.v1",
        "model": args.model,
        "seed": args.seed,
        "status": "ENGINEERING_TIMEOUT" if timed_out else ("COMPLETED" if process.returncode == 0 else "FAILED"),
        "return_code": process.returncode,
        "runtime_seconds": elapsed,
        "timeout_seconds": args.timeout_seconds,
        "command": command,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "execution_artifacts_eligible_for_scientific_audit": not timed_out and process.returncode == 0,
        "scientific_conclusion_established": False,
    }
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{args.model}-S{args.seed}-supervisor.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 124 if timed_out else int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
