from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_DATASET_SHA256 = "d52c1399b844165f94fc156fc7919be9fbb0bf214dfff74b5c48bf429917759e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--dataset", default="resources/data/shortperiod_new.csv")
    parser.add_argument("--output", default="outputs/teammate/environment.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = (root / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset).resolve()
    versions = {}
    errors = []
    for module_name in ("numpy", "pandas", "sklearn", "joblib", "pydantic", "torch"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "UNKNOWN"))
        except Exception as exc:
            versions[module_name] = None
            errors.append(f"dependency_unavailable:{module_name}:{type(exc).__name__}")
    dataset_hash = _sha256(dataset) if dataset.is_file() else None
    if dataset_hash != EXPECTED_DATASET_SHA256:
        errors.append(f"dataset_sha256_mismatch:{dataset_hash}")
    if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
        errors.append(f"python_3_11_required:{platform.python_version()}")
    payload = {
        "schema_version": "boilermind.teammate_environment.v1",
        "operator_id": args.operator_id,
        "machine_id": args.machine_id,
        "project_root": str(root),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "dependency_versions": versions,
        "dataset_path": str(dataset),
        "dataset_size_bytes": dataset.stat().st_size if dataset.is_file() else None,
        "dataset_sha256": dataset_hash,
        "expected_dataset_sha256": EXPECTED_DATASET_SHA256,
        "status": "PASSED" if not errors else "FAILED",
        "issues": errors,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
