from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boilermind.core.contracts import ExperimentContract, ExperimentPlan
from boilermind.experiment.time_series_data import TimeSeriesDataset
from boilermind.experiment.unified_runner import ExperimentExecutionError, UnifiedExperimentRunner
from boilermind.models.execution_environment import ExecutionEnvironment
from boilermind.planning.plan_contracts import ExperimentCapabilitySnapshot
from boilermind.planning.plan_gate import compile_plan_to_contract


METRICS = {"mae_t_h": 1.0, "rmse_t_h": 1.0, "r2": 0.0, "mbe_t_h": 0.0}


class FakeAdapter:
    def __init__(self, name, calls, fail=False, torch=False):
        self.name, self.calls, self.fail = name, calls, fail
        self.warning_messages, self.config, self.params = [], {}, {"alpha": 1.0}
        self.runtime_seconds, self.device = .01, "cpu"
        self.epochs_completed, self.best_epoch = (1, 1) if torch else (None, None)
        self.training_loss, self.validation_loss = (.1, .2) if torch else (None, None)
        self.target_scaling_method = "standard_train_only" if torch else None
        self.target_mean_ = np.array([0.0]) if torch else None
        self.target_scale_ = np.array([1.0]) if torch else None
        self.estimator = SimpleNamespace(name=name)
        self.model = SimpleNamespace(state_dict=lambda: {"weight": [1.0]})
        self._torch = SimpleNamespace(save=lambda state, path: Path(path).write_bytes(b"checkpoint"))

    def fit(self, X, y, **kwargs):
        self.calls.append((self.name, "fit", tuple(X.shape), kwargs))
        if self.fail:
            raise RuntimeError("intentional_failure")
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def evaluate(self, y, prediction):
        return dict(METRICS)

    def _load_torch(self):
        return self._torch


class FakeRegistry:
    def __init__(self, frameworks, fail=()):
        self.frameworks, self.fail, self.calls = frameworks, set(fail), []

    def get(self, name):
        framework = "heuristic" if name == "persistence" else self.frameworks[name]
        return SimpleNamespace(model_name=name, framework=framework,
            required_input_type="sequence_window" if framework == "torch" else "flattened_window",
            checkpoint_path=None, checkpoint_compatibility={"compatible": False})

    def build_adapter(self, name, **kwargs):
        return FakeAdapter(name, self.calls, name in self.fail, self.frameworks.get(name) == "torch")


class FakeBuilder:
    def build_from_csv(self, path, contract):
        X = np.zeros((4, contract.window_steps, 30))
        y = np.arange(4, dtype=float)[:, None]
        return TimeSeriesDataset(X, y, X.copy(), y.copy(), X.copy(), y.copy(),
            y.copy(), y.copy(), y.copy(), None,
            {"train": np.arange(4), "validation": np.arange(4), "locked_test": np.arange(4)},
            {"train": np.arange(4) + 1, "validation": np.arange(4) + 1, "locked_test": np.arange(4) + 1})


def make_runner(tmp_path, frameworks, fail=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "data.csv"
    dataset.write_text("fixture", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    registry = FakeRegistry(frameworks, fail)
    capability = SimpleNamespace(dataset_path=dataset, sampling_interval_seconds=15,
        window_steps=4, prediction_horizon_steps=2, train_ratio=.6, validation_ratio=.2,
        available_models=lambda: list(frameworks))
    environment = ExecutionEnvironment(os="test", python_version="3.11", sklearn_available=True,
        torch_available=True, cuda_available=False, gpu_available=False)
    runner = UnifiedExperimentRunner(capability_registry=capability, model_registry=registry,
        dataset_builder=FakeBuilder(), environment=environment, output_dir=tmp_path / "experiments")
    return runner, registry, digest


def make_contract(models, digest, **updates):
    values = dict(experiment_id="EXP-U", problem_id="P", hypothesis_id="H", plan_id="PL",
        dataset_id="D", dataset_hash=digest, input_variables=["feature_01"], target_variable="main_steam_mass_flow",
        train_split="train", validation_split="validation", test_split="locked", baseline_models=["persistence"],
        reference_models=["persistence"], candidate_models=list(models), metrics=["MAE"],
        confirmation_criteria=["c"], falsification_criteria=["f"], window_steps=4,
        prediction_horizon_steps=2, sampling_interval_seconds=15)
    values.update(updates)
    return ExperimentContract(**values)


@pytest.mark.parametrize("frameworks", [
    {"ridge": "sklearn"}, {"lstm": "torch"},
    {"ridge": "sklearn", "lstm": "torch", "transformer": "torch"},
])
def test_framework_routing_and_frozen_model_set(tmp_path, frameworks):
    runner, registry, digest = make_runner(tmp_path, frameworks)
    contract = make_contract(frameworks, digest, max_epochs=1)
    result, _trace = runner.run(contract)
    assert list(result.model_records) == list(contract.candidate_models)
    assert [item[0] for item in registry.calls if item[1] == "fit"] == list(contract.candidate_models)
    for name, framework in frameworks.items():
        expected = (4, 4, 30) if framework == "torch" else (4, 120)
        assert next(item[2] for item in registry.calls if item[0] == name) == expected
        assert (
            tmp_path
            / "experiments"
            / "EXP-U"
            / "predictions"
            / f"{name}_locked_test_predictions.csv"
        ).is_file()


def test_failure_no_fallback_and_allow_partial_failure(tmp_path):
    frameworks = {"ridge": "sklearn", "lstm": "torch"}
    runner, registry, digest = make_runner(tmp_path, frameworks, fail={"lstm"})
    with pytest.raises(ExperimentExecutionError) as caught:
        runner.run(make_contract(frameworks, digest))
    assert set(caught.value.result.model_records) == set(frameworks)
    assert caught.value.result.model_records["lstm"].fit_success is False
    assert [item[0] for item in registry.calls if item[1] == "fit"] == ["ridge", "lstm"]

    runner, _registry, digest = make_runner(tmp_path / "partial", frameworks, fail={"lstm"})
    result, _ = runner.run(make_contract(frameworks, digest, allow_partial_failure=True))
    assert result.model_records["ridge"].fit_success is True
    assert result.model_records["lstm"].fit_success is False


def test_model_selection_uses_canonical_mae_for_volume_metric_aliases(
    tmp_path, monkeypatch
):
    values = {
        "ridge": {"mae_m3_s": 2.0, "rmse_m3_s": 3.0, "r2": 0.1, "mbe_m3_s": 0.2},
        "bayesianridge": {
            "mae_m3_s": 1.0, "rmse_m3_s": 2.0, "r2": 0.3, "mbe_m3_s": 0.1,
        },
        "persistence": {
            "mae_m3_s": 4.0, "rmse_m3_s": 5.0, "r2": 0.0, "mbe_m3_s": 0.3,
        },
    }

    monkeypatch.setattr(
        FakeAdapter,
        "evaluate",
        lambda self, _truth, _prediction: dict(values[self.name]),
    )
    runner, _registry, digest = make_runner(
        tmp_path, {"ridge": "sklearn", "bayesianridge": "sklearn"}
    )
    result, _trace = runner.run(
        make_contract(["ridge", "bayesianridge"], digest)
    )

    assert result.metrics["MAE"] == 1.0
    assert result.metrics["mae_m3_s"] == 1.0
    assert result.baseline_metrics["MAE"] == 4.0


def test_regime_operation_and_experiment_type_must_match(tmp_path):
    runner, _registry, digest = make_runner(tmp_path, {"ridge": "sklearn"})
    with pytest.raises(ValueError, match="regime_operation_contract_mismatch"):
        runner.run(
            make_contract(
                ["ridge"],
                digest,
                experiment_type="model_comparison",
                required_operations=["regime_stratified_evaluation"],
            )
        )


def test_unavailable_model_fails_closed_and_manifest_provenance(tmp_path):
    runner, _registry, digest = make_runner(tmp_path, {"lstm": "torch"})
    runner.capability.available_models = lambda: []
    with pytest.raises(ExperimentExecutionError) as caught:
        runner.run(make_contract(["lstm"], digest))
    assert "planned_model_not_currently_executable" in caught.value.result.model_records["lstm"].failure_reason
    manifest = tmp_path / "experiments" / "EXP-U" / "manifest.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert (payload["problem_id"], payload["hypothesis_id"], payload["plan_id"], payload["experiment_id"]) == ("P", "H", "PL", "EXP-U")
    assert payload["dataset_sha256"] == digest


def test_plan_contract_preserves_models_policy_constraints_and_ids(tmp_path):
    plan = ExperimentPlan(plan_id="PL", problem_id="P", hypothesis_id="H", objective="o",
        experimental_design="d", baseline_description="b", intervention_description="i",
        required_variables=["feature_01"], target="main_steam_mass_flow", metrics=["MAE"],
        expected_observation="e", confirmation_criteria=["c"], falsification_criteria=["f"],
        candidate_models=["ridge", "lstm"], reference_models=["persistence"], current_executable=True,
        prediction_horizon_steps=40, sampling_interval_seconds=15, required_operations=["model_comparison"],
        hard_constraints=["no_leakage"], allow_partial_failure=True, max_epochs=2, allowed_devices=["cpu"])
    snapshot = ExperimentCapabilitySnapshot(snapshot_id="S", dataset_id="D", dataset_hash="hash",
        available_variables=["feature_01"], available_target_variables=["main_steam_mass_flow"],
        available_baseline_models=["persistence"], available_candidate_models=["ridge", "lstm"],
        available_metrics=["MAE"], train_split="train", validation_split="validation", test_split="locked",
        data_frozen=True, leakage_policy_verified=True, prediction_horizon_steps=40, sampling_interval_seconds=15)
    contract, report = compile_plan_to_contract(plan, snapshot, target_variable=plan.target,
        baseline_models=list(plan.reference_models), candidate_models=list(plan.candidate_models))
    assert report.passed and contract.problem_id == "P" and contract.hypothesis_id == "H" and contract.plan_id == "PL"
    assert contract.candidate_models == plan.candidate_models
    assert contract.required_operations == plan.required_operations and contract.constraints == plan.hard_constraints
    assert contract.allow_partial_failure is True and contract.max_epochs == 2
    rejected, report = compile_plan_to_contract(plan, snapshot, target_variable=plan.target,
        baseline_models=list(plan.reference_models), candidate_models=["ridge"])
    assert rejected is None and "candidate_models_must_come_from_plan" in report.issues
