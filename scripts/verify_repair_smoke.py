from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    y_true = np.asarray([float(row["y_true"]) for row in rows])
    y_pred = np.asarray([float(row["y_pred"]) for row in rows])
    error = y_pred - y_true
    return {
        "mae_t_h": float(np.mean(np.abs(error))),
        "rmse_t_h": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "mbe_t_h": float(np.mean(error)),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    record = result["model_records"][args.model]
    checks = {}
    for split, expected_name in (("validation", "validation_metrics"), ("locked_test", "locked_test_metrics")):
        path = root / "predictions" / f"{args.model}_{split}_predictions.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        calculated = _metrics(rows)
        expected = record[expected_name]
        metric_match = all(abs(calculated[name] - expected[name]) <= args.tolerance for name in expected)
        source = np.asarray([int(row["source_index"]) for row in rows])
        target = np.asarray([int(row["target_index"]) for row in rows])
        checks[split] = {
            "rows": len(rows),
            "metric_match": metric_match,
            "indices_strictly_increasing": bool(np.all(np.diff(source) > 0) and np.all(np.diff(target) > 0)),
            "horizon_alignment_constant": bool(np.all(target - source == target[0] - source[0])),
            "calculated": calculated,
        }
    checks["split_separation"] = bool(
        max(int(row["target_index"]) for row in csv.DictReader(
            (root / "predictions" / f"{args.model}_validation_predictions.csv").open("r", encoding="utf-8", newline="")
        )) < min(int(row["target_index"]) for row in csv.DictReader(
            (root / "predictions" / f"{args.model}_locked_test_predictions.csv").open("r", encoding="utf-8", newline="")
        ))
    )
    passed = all(item["metric_match"] and item["indices_strictly_increasing"] and item["horizon_alignment_constant"]
                 for item in (checks["validation"], checks["locked_test"])) and checks["split_separation"]
    payload = {"model": args.model, "passed": passed, "checks": checks}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
