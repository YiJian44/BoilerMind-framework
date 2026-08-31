from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.experiment.unified_runner import UnifiedExperimentRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = PROJECT_ROOT / "resources" / "data" / "shortperiod_new.csv"
    with tempfile.TemporaryDirectory(prefix="boilermind_feature_intervention_") as directory:
        subset = Path(directory) / "chronological_prefix.csv"
        pd.read_csv(source, header=None).iloc[:256].to_csv(subset, header=False, index=False)
        digest = hashlib.sha256(subset.read_bytes()).hexdigest()
        capability = ExperimentCapabilityRegistry(dataset_path=subset)
        contract = ExperimentContract(
            experiment_id="EXP-FEATURE-INTERVENTION-SMOKE", problem_id="SMOKE-PROBLEM",
            hypothesis_id="SMOKE-FEATURE-HYPOTHESIS", plan_id="SMOKE-FEATURE-PLAN",
            experiment_type="feature_intervention", dataset_id=capability.dataset_id(), dataset_hash=digest,
            input_variables=[f"feature_{index:02d}" for index in range(1, 9)],
            target_variable="main_steam_mass_flow", train_split="train",
            validation_split="validation", test_split="locked_test",
            baseline_models=["persistence"], reference_models=["persistence"], candidate_models=["ridge"],
            control={"model": "ridge", "features": [f"feature_{index:02d}" for index in range(1, 6)]},
            treatment={"model": "ridge", "features": [f"feature_{index:02d}" for index in range(1, 9)]},
            metrics=["MAE", "RMSE", "R2"], confirmation_criteria=["delta_MAE_lt_0"],
            falsification_criteria=["delta_MAE_ge_0"], sampling_interval_seconds=15,
            window_steps=20, prediction_horizon_steps=40, random_seed=42,
        )
        result, _trace = UnifiedExperimentRunner(capability_registry=capability,
            output_dir=PROJECT_ROOT / "outputs" / "experiments").run(contract)
        print("control", result.control_metrics)
        print("treatment", result.treatment_metrics)
        print("deltas", result.metric_deltas)
        print("valid", result.experiment_valid)


if __name__ == "__main__":
    main()
