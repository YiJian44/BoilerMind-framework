from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.experiment.unified_runner import UnifiedExperimentRunner
from boilermind.models.execution_environment import ExecutionEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "resources" / "data" / "shortperiod_new.csv"


def main() -> None:
    environment = ExecutionEnvironment.detect()
    if not environment.torch_available or environment.cuda_available:
        raise RuntimeError("cpu_only_torch_environment_required")

    with tempfile.TemporaryDirectory(prefix="boilermind_torch_smoke_") as directory:
        subset = Path(directory) / "chronological_prefix.csv"
        pd.read_csv(SOURCE_DATASET, header=None).iloc[:256].to_csv(subset, header=False, index=False)
        digest = hashlib.sha256(subset.read_bytes()).hexdigest()
        capability = ExperimentCapabilityRegistry(dataset_path=subset, environment=environment)
        contract = ExperimentContract(
            experiment_id="EXP-TORCH-CPU-SMOKE", problem_id="SMOKE-PROBLEM",
            hypothesis_id="SMOKE-HYPOTHESIS", plan_id="SMOKE-PLAN",
            dataset_id=capability.dataset_id(), dataset_hash=digest,
            input_variables=[f"feature_{index:02d}" for index in range(1, 31)],
            target_variable="main_steam_mass_flow", train_split="train",
            validation_split="validation", test_split="locked_test",
            baseline_models=["persistence"], reference_models=["persistence"],
            candidate_models=["lstm", "gru", "transformer", "dlinear"],
            metrics=["MAE", "RMSE", "R2", "MBE"],
            confirmation_criteria=["smoke_execution_only"],
            falsification_criteria=["any_model_runtime_failure"],
            sampling_interval_seconds=15, window_steps=20,
            prediction_horizon_steps=40, random_seed=42, max_epochs=1,
            allowed_devices=["cpu"], allow_partial_failure=False,
        )
        runner = UnifiedExperimentRunner(capability_registry=capability, environment=environment,
            output_dir=PROJECT_ROOT / "outputs" / "experiments")
        result, _trace = runner.run(contract)
        for name, record in result.model_records.items():
            print(name, record.fit_success, record.device, record.epochs_completed,
                  record.validation_metrics.get("mae_t_h"))


if __name__ == "__main__":
    main()
