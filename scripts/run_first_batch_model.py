from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.experiment.unified_runner import UnifiedExperimentRunner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="outputs/first_batch/model_runs",
    )
    args = parser.parse_args()

    capability = ExperimentCapabilityRegistry()
    if args.model not in capability.available_models():
        raise SystemExit(f"model_not_currently_executable:{args.model}")

    dataset_path = Path(capability.dataset_path)
    dataset_hash = _sha256(dataset_path)
    experiment_id = f"BM-SEED-01-{args.model.upper()}-S{args.seed}"
    contract = ExperimentContract(
        experiment_id=experiment_id,
        problem_id="BM-P-MASS-H40-MODEL-COMPARISON",
        hypothesis_id="H-SEED-01",
        plan_id="PL-SEED-01-UNIFIED-BENCHMARK",
        experiment_type="model_comparison",
        primary_metric="MAE",
        secondary_metrics=["RMSE", "R2", "MBE"],
        reference_models=["persistence"],
        prediction_horizon_steps=40,
        sampling_interval_seconds=15,
        window_steps=20,
        locked_test_used_for_selection=False,
        required_operations=[
            "model_comparison",
            "reference_model_comparison",
            "chronological_validation",
            "locked_test_evaluation",
        ],
        constraints=[
            "chronological_split",
            "validation_only_model_selection",
            "locked_test_not_used_for_selection",
            "same_dataset_and_sample_indices",
        ],
        allow_partial_failure=True,
        max_epochs=args.max_epochs,
        allowed_devices=["cpu"],
        dataset_id=capability.dataset_id(),
        dataset_hash=dataset_hash,
        input_variables=capability.available_variables(),
        target_variable="main_steam_mass_flow",
        train_split="chronological_0_70",
        validation_split="chronological_70_85",
        test_split="chronological_85_100_locked",
        baseline_models=["persistence"],
        candidate_models=[args.model],
        metrics=["MAE", "RMSE", "R2", "MBE"],
        confirmation_criteria=[
            "candidate_locked_test_MAE_lower_than_persistence",
        ],
        falsification_criteria=[
            "candidate_locked_test_MAE_not_lower_than_persistence",
        ],
        random_seed=args.seed,
    )
    runner = UnifiedExperimentRunner(
        capability_registry=capability,
        output_dir=Path(args.output_dir),
    )
    result, trace = runner.run(contract)
    record = result.model_records[args.model]
    summary = {
        "experiment_id": experiment_id,
        "model": args.model,
        "seed": args.seed,
        "fit_success": record.fit_success,
        "fit_converged": record.fit_converged,
        "failure_reason": record.failure_reason,
        "warnings": record.warnings,
        "runtime_seconds": record.runtime_seconds,
        "validation_metrics": record.validation_metrics,
        "locked_test_metrics": record.locked_test_metrics,
        "persistence_locked_test_metrics": result.baseline_metrics,
        "dataset_sha256": dataset_hash,
        "trace": {
            "dataset_frozen": trace.dataset_frozen,
            "leakage_check_passed": trace.leakage_check_passed,
            "baseline_valid": trace.baseline_valid,
            "metric_check_passed": trace.metric_check_passed,
            "notes": trace.notes,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if record.fit_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
