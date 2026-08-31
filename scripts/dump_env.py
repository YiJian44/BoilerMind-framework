"""Dump the runtime environment for the 31V model-library audit package.

Run on the training host (server) so the environment report matches the
actual training environment. Writes environment/{python_version.txt,
pip_freeze.txt, torch_environment.json, hardware.json}.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(errors="replace").splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _pip_freeze() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        return f"# pip freeze failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "environment"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "python_version.txt").write_text(
        f"python: {sys.version}\nplatform: {platform.platform()}\n", encoding="utf-8"
    )
    (out / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")

    torch_env: dict = {"seed": args.seed, "deterministic_algorithms": False}
    try:
        import torch

        torch_env.update({
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.version.cuda else None,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_count": int(torch.cuda.device_count()),
            "seed_notes": "torch.manual_seed(seed) and np.random.seed(seed) set in TorchSensor.fit",
        })
    except Exception as exc:
        torch_env["error"] = str(exc)
    (out / "torch_environment.json").write_text(
        json.dumps(torch_env, indent=2), encoding="utf-8"
    )

    hardware = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu": _cpu_name(),
        "memory_bytes": _memory_bytes(),
    }
    (out / "hardware.json").write_text(
        json.dumps(hardware, indent=2), encoding="utf-8"
    )

    print(f"environment dump -> {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
