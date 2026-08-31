from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        default="outputs/first_batch/model_runs",
    )
    parser.add_argument(
        "--output",
        default="outputs/first_batch/BM-SEED-01-summary.json",
    )
    args = parser.parse_args()

    rows = []
    for result_path in sorted(Path(args.runs_dir).glob("BM-SEED-01-*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        model, record = next(iter(payload["model_records"].items()))
        baseline = payload["baseline_metrics"]
        locked = record["locked_test_metrics"]
        base_mae = baseline.get("mae_t_h")
        model_mae = locked.get("mae_t_h")
        improvement = None
        if base_mae and model_mae is not None:
            improvement = (base_mae - model_mae) / base_mae
        rows.append({
            "experiment_id": payload["experiment_id"],
            "model": model,
            "fit_success": record["fit_success"],
            "fit_converged": record["fit_converged"],
            "runtime_seconds": record["runtime_seconds"],
            "validation_metrics": record["validation_metrics"],
            "locked_test_metrics": locked,
            "persistence_locked_test_metrics": baseline,
            "locked_test_mae_improvement_ratio": improvement,
            "warnings": record["warnings"],
            "failure_reason": record["failure_reason"],
            "result_path": str(result_path.resolve()),
        })

    rows.sort(key=lambda row: row["locked_test_metrics"].get("mae_t_h", float("inf")))
    output = {
        "schema_version": "boilermind.first_batch_summary.v1",
        "problem_id": "BM-P-MASS-H40-MODEL-COMPARISON",
        "protocol": {
            "target": "main_steam_mass_flow",
            "window_steps": 20,
            "prediction_horizon_steps": 40,
            "sampling_interval_seconds": 15,
            "split": [0.70, 0.15, 0.15],
            "selection_scope": "validation_only",
            "locked_test_used_for_selection": False,
            "seed": 42,
        },
        "completed_model_count": len(rows),
        "planned_model_count": 14,
        "not_completed": [
            {
                "model": "elasticnet",
                "status": "ENGINEERING_TIMEOUT",
                "reason": "fit_did_not_complete_within_first_batch_operational_budget",
            },
            {
                "model": "gpr",
                "status": "RESOURCE_SAFETY_BLOCKED",
                "reason": "exact_gpr_not_launched_on_approximately_17600_training_windows_without_hard_memory_and_time_isolation",
            },
        ],
        "ranking_by_locked_test_mae": rows,
        "scientific_validity_notes": [
            "fit_success_does_not_by_itself_establish_scientific_validity",
            "torch_models_with_extreme_error_require_training_and_target_scaling_audit_before scientific interpretation",
            "predictions_were_not_persisted_by_the_current_unified_runner_so_regime_analysis_requires a runner fix or reproducible re-evaluation",
            "non_convergence_and environment warnings must be separated",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
