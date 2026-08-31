from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_metrics(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    y_true = np.asarray([float(row["y_true"]) for row in rows])
    y_pred = np.asarray([float(row["y_pred"]) for row in rows])
    source = np.asarray([int(row["source_index"]) for row in rows])
    target = np.asarray([int(row["target_index"]) for row in rows])
    error = y_pred - y_true
    metrics = {
        "mae_m3_s": float(np.mean(np.abs(error))),
        "rmse_m3_s": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "mbe_m3_s": float(np.mean(error)),
    }
    checks = {
        "count": len(rows),
        "source_strictly_increasing": bool(np.all(np.diff(source) > 0)),
        "target_strictly_increasing": bool(np.all(np.diff(target) > 0)),
        "horizon": int(target[0] - source[0]),
        "horizon_constant": bool(np.all(target - source == target[0] - source[0])),
        "dataset_hashes": sorted({row["dataset_sha256"] for row in rows}),
        "protocol_hashes": sorted({row["protocol_sha256"] for row in rows}),
    }
    return metrics, checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root")
    parser.add_argument("--metric-tolerance", type=float, default=2e-6)
    parser.add_argument("--skip-full-hash", action="store_true")
    args = parser.parse_args()
    root = Path(args.package_root).resolve()
    package_manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    library = json.loads((root / "model_library/model_library.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    hash_rows = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    if not args.skip_full_hash:
        for row in hash_rows:
            path = root / row["path"]
            if not path.is_file():
                failures.append(f"missing:{row['path']}")
            elif path.stat().st_size != int(row["size_bytes"]):
                failures.append(f"size_mismatch:{row['path']}")
            elif _sha256(path) != row["sha256"]:
                failures.append(f"sha256_mismatch:{row['path']}")
    dataset = root / "data/boiler_181var_clean.csv"
    if _sha256(dataset) != package_manifest["dataset_sha256"]:
        failures.append("dataset_sha256_mismatch")
    by_id = {item["id"]: item for item in library["models"]}
    prediction_checks = {}
    max_metric_difference = 0.0
    for horizon in (40, 80):
        protocol_hash = package_manifest[f"protocol_sha256_h{horizon}"]
        manifest_path = root / "model_library" / "manifests" / "31v_direct" / f"h{horizon}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_counts = {
            split: int(manifest["split"][split]["count"])
            for split in ("validation", "locked_test")
        }
        for model in ("persistence", "ridge", "bayesianridge", "elasticnet", "pls", "svr",
                      "rf", "mlp", "knn", "hgb", "transformer", "lstm", "dlinear", "gru"):
            entry = by_id[f"{model}_h{horizon}"]
            for split, expected_count in split_counts.items():
                path = root / f"model_library/predictions/31v_direct/h{horizon}/{model}_{split}_predictions.csv"
                if not path.is_file():
                    failures.append(f"prediction_missing:h{horizon}:{model}:{split}")
                    continue
                calculated, checks = _prediction_metrics(path)
                key = f"h{horizon}:{model}:{split}"
                prediction_checks[key] = checks
                if checks["count"] != expected_count:
                    failures.append(f"prediction_count:{key}:{checks['count']}")
                if not checks["source_strictly_increasing"] or not checks["target_strictly_increasing"]:
                    failures.append(f"prediction_indices:{key}")
                if not checks["horizon_constant"] or checks["horizon"] != horizon:
                    failures.append(f"prediction_horizon:{key}")
                if checks["dataset_hashes"] != [package_manifest["dataset_sha256"]]:
                    failures.append(f"prediction_dataset_hash:{key}")
                if checks["protocol_hashes"] != [protocol_hash]:
                    failures.append(f"prediction_protocol_hash:{key}")
                for metric, expected in entry["metrics"][split].items():
                    difference = abs(calculated[metric] - float(expected))
                    max_metric_difference = max(max_metric_difference, difference)
                    if not math.isfinite(difference) or difference > args.metric_tolerance:
                        failures.append(f"metric_mismatch:{key}:{metric}:{difference}")
            weights = entry.get("weights") or {}
            if model != "persistence" and weights.get("exists"):
                model_dir = root / "model_library" / weights["dir"]
                for filename in (weights.get("weight_file"), weights.get("scaler")):
                    if filename and not (model_dir / filename).is_file():
                        failures.append(f"weight_artifact_missing:h{horizon}:{model}:{filename}")
    report = {
        "schema_version": "boilermind.31v_artifact_verification.v1",
        "package_root": str(root),
        "declared_hash_file_count": len(hash_rows),
        "full_hash_checked": not args.skip_full_hash,
        "library_model_count": len(library["models"]),
        "prediction_file_count_checked": len(prediction_checks),
        "max_metric_absolute_difference": max_metric_difference,
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
