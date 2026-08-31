from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.experiment.unified_runner import UnifiedExperimentRunner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", default="outputs/teammate/model_runs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    task_bundle = json.loads(Path(args.task_file).resolve().read_text(encoding="utf-8"))
    task = next((item for item in task_bundle.get("tasks", []) if item.get("task_id") == args.task_id), None)
    if task is None:
        raise SystemExit(f"unknown_task_id:{args.task_id}")
    if task.get("execution_authorization") != "SMOKE_APPROVED":
        raise SystemExit(f"task_not_authorized:{args.task_id}")
    dataset = (root / task_bundle["dataset_relative_path"]).resolve()
    digest = _sha256(dataset)
    if digest != task_bundle["dataset_sha256"]:
        raise SystemExit(f"dataset_sha256_mismatch:{digest}")
    os.environ["BOILERMIND_REAL_DATASET_PATH"] = str(dataset)
    capability = ExperimentCapabilityRegistry(dataset_path=dataset)
    model = task["model"]
    contract = ExperimentContract(
        experiment_id=task["task_id"], problem_id="BM-P-MASS-H40-MODEL-COMPARISON",
        hypothesis_id="H-SMOKE-02", plan_id="PL-SMOKE-02", experiment_type="model_comparison",
        primary_metric="MAE", secondary_metrics=["RMSE", "R2", "MBE"],
        reference_models=["persistence"], prediction_horizon_steps=40,
        sampling_interval_seconds=15, window_steps=20, locked_test_used_for_selection=False,
        required_operations=["model_comparison", "reference_model_comparison", "chronological_validation", "locked_test_evaluation"],
        constraints=["chronological_split", "validation_only_model_selection", "locked_test_not_used_for_selection", "same_dataset_and_sample_indices"],
        allow_partial_failure=True, max_epochs=int(task["max_epochs"]), allowed_devices=["cpu"],
        dataset_id=capability.dataset_id(), dataset_hash=digest,
        input_variables=capability.available_variables(), target_variable="main_steam_mass_flow",
        train_split="chronological_0_70", validation_split="chronological_70_85",
        test_split="chronological_85_100_locked", baseline_models=["persistence"],
        candidate_models=[model], metrics=["MAE", "RMSE", "R2", "MBE"],
        confirmation_criteria=["candidate_locked_test_MAE_lower_than_persistence"],
        falsification_criteria=["candidate_locked_test_MAE_not_lower_than_persistence"],
        random_seed=int(task["seed"]),
    )
    result, trace = UnifiedExperimentRunner(capability_registry=capability, output_dir=root / args.output_dir).run(contract)
    record = result.model_records[model]
    print(json.dumps({"task_id": task["task_id"], "fit_success": record.fit_success,
                      "fit_converged": record.fit_converged, "locked_test_metrics": record.locked_test_metrics,
                      "trace": trace.__dict__}, ensure_ascii=False, indent=2, default=str))
    return 0 if record.fit_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
