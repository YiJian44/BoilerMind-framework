from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.time_series_data import DatasetBuilder
from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
    ExperimentCapabilityRegistry,
)
from boilermind.experiment.unified_runner import UnifiedExperimentRunner
from boilermind.core.contracts import ResearchProblemSpec
from boilermind.models import build_default_registry
from boilermind.models.execution_environment import ExecutionEnvironment
from boilermind.skills.planning_skill import PlanningSkill


def _contract(path: Path, digest: str, **updates) -> ExperimentContract:
    values = dict(
        experiment_id="EXP-31V", problem_id="P", hypothesis_id="H", plan_id="PL",
        dataset_id="D31V", dataset_hash=digest,
        input_variables=["feature_1"], target_variable="steam_volumetric_flow",
        train_split="train", validation_split="validation", test_split="locked_test",
        baseline_models=["persistence"], candidate_models=["ridge"], metrics=["MAE"],
        confirmation_criteria=["c"], falsification_criteria=["f"],
        window_steps=20, prediction_horizon_steps=40, sampling_interval_seconds=15,
        execution_requirements={"dataset_path": str(path)},
    )
    values.update(updates)
    return ExperimentContract(**values)


def _dataset(path: Path) -> str:
    columns = {str(index): np.linspace(index, index + 1, 180) for index in range(1, 182)}
    columns["1"] = np.linspace(15.0, 16.0, 180)
    columns["9"] = np.linspace(530.0, 540.0, 180)
    columns["16"] = np.linspace(800.0, 900.0, 180)
    pd.DataFrame(columns).to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_volume_uses_same_runner_with_31_features_and_derived_target(tmp_path):
    path = tmp_path / "boiler.csv"
    digest = _dataset(path)
    runner = object.__new__(UnifiedExperimentRunner)
    runner.dataset_builder = DatasetBuilder()

    actual_path, actual_hash, data = runner._prepare(_contract(path, digest))

    assert actual_path == path
    assert actual_hash == digest
    assert data.X_train.shape[1:] == (20, 31)
    assert data.y_train.ndim == 2
    assert float(data.y_train.min()) > 0


@pytest.mark.parametrize("window_steps", [10, 20, 40, 80])
def test_direct_volume_supported_windows_build_contract_sequences(
    tmp_path, window_steps,
):
    path = tmp_path / "boiler.csv"
    digest = _dataset(path)
    runner = object.__new__(UnifiedExperimentRunner)
    runner.dataset_builder = DatasetBuilder()

    _, _, data = runner._prepare(
        _contract(path, digest, window_steps=window_steps)
    )

    assert data.X_train.shape[1:] == (window_steps, 31)


@pytest.mark.parametrize("window_steps", [10, 20, 40, 80])
def test_direct_volume_supported_windows_execute_real_ridge(
    tmp_path, window_steps,
):
    path = tmp_path / "boiler.csv"
    digest = _dataset(path)
    capability = DirectVolume31VCapabilityRegistry(
        dataset_path=path,
        enabled_models=("ridge",),
    )
    runner = UnifiedExperimentRunner(
        capability_registry=capability,
        output_dir=tmp_path / "experiments",
    )

    result, trace = runner.run(
        _contract(
            path,
            digest,
            experiment_id=f"EXP-W{window_steps}",
            window_steps=window_steps,
        )
    )

    assert result.status.value == "completed"
    assert result.model_records["ridge"].fit_success is True
    assert trace.leakage_check_passed is True


@pytest.mark.parametrize("field,value,error", [
    ("prediction_horizon_steps", 60, "direct_volume_horizon_not_supported"),
    ("window_steps", 30, "direct_volume_window_steps_not_supported:30"),
    ("sampling_interval_seconds", 30, "direct_volume_sampling_interval_must_equal_15"),
])
def test_direct_volume_protocol_is_fail_closed(tmp_path, field, value, error):
    path = tmp_path / "boiler.csv"
    digest = _dataset(path)
    runner = object.__new__(UnifiedExperimentRunner)
    runner.dataset_builder = DatasetBuilder()
    with pytest.raises(ValueError, match=error):
        runner._prepare(_contract(path, digest, **{field: value}))


def test_direct_volume_rejects_models_outside_verified_set():
    runner = object.__new__(UnifiedExperimentRunner)
    runner.model_registry = SimpleNamespace(
        get=lambda _name: SimpleNamespace(framework="sklearn")
    )
    runner.backend_resolver = SimpleNamespace(resolve=lambda _spec: "sklearn")
    with pytest.raises(RuntimeError, match="direct_volume_model_not_verified"):
        runner._execute_model(
            "gpr", _contract(Path("unused"), "real"), None, "hash", None, Path("unused")
        )


def _problem(target: str) -> ResearchProblemSpec:
    return ResearchProblemSpec(
        problem_id="P", original_question="预测未来体积流量",
        research_object="锅炉", target_variable=target,
        operating_condition="全工况", research_goal="验证可预测性",
    )


def test_direct_volume_planning_pool_intersects_catalog_and_real_environment():
    environment = ExecutionEnvironment(
        os="test", python_version="3.11", sklearn_available=True,
        torch_available=False, cuda_available=False,
        gpu_available=False,
    )
    capability = DirectVolume31VCapabilityRegistry(environment=environment)
    pool = build_default_registry().match_task_capability(
        task_type="prediction", target_variable="steam_volumetric_flow",
        metrics=["MAE"], capability=capability,
    )
    names = {spec.model_name for spec in pool}
    assert {"ridge", "bayesianridge", "rf", "persistence"} <= names
    assert "lstm" not in capability.available_models()


def test_direct_volume_capability_declares_supported_windows():
    capability = DirectVolume31VCapabilityRegistry()
    assert capability.window_steps == 20
    assert capability.supported_window_steps == (10, 20, 40, 80)
    assert capability.snapshot()["supported_window_steps"] == [10, 20, 40, 80]


def test_direct_volume_default_is_not_overridden_by_generic_dataset_env(
    tmp_path, monkeypatch,
):
    unrelated = tmp_path / "mass-flow.csv"
    unrelated.write_text("1,2\n", encoding="utf-8")
    monkeypatch.setenv("BOILERMIND_REAL_DATASET_PATH", str(unrelated))

    capability = DirectVolume31VCapabilityRegistry(enabled_models=("ridge",))

    assert capability.dataset_path == capability.DEFAULT_DATASET_PATH


def test_problem_horizon_conflict_with_hypothesis_fails_closed():
    capability = DirectVolume31VCapabilityRegistry()
    hypothesis = {
        "hypothesis_id": "H-PROBLEM-MODELS",
        "hypothesis": "比较 Ridge 与 Transformer 的 h40 预测误差",
        "verification_intent": "执行模型比较",
        "falsification_condition": "候选模型排序与假设不一致",
        "variables": ["steam_volumetric_flow"],
    }
    problem = _problem("steam_volumetric_flow").model_dump(mode="json")
    problem.update({
        "required_models": ["ridge", "bayesianridge", "rf", "lstm", "transformer"],
        "reference_models": ["persistence"],
        "required_horizon_steps": 80,
        "required_operations": ["model_comparison", "locked_test_evaluation"],
        "protocol_constraints": ["locked_test_not_used_for_selection"],
        "metrics": ["MAE", "RMSE", "R2"],
    })
    output = PlanningSkill(capability_registry=capability).execute({
        "problem_id": problem["problem_id"], "research_problem": problem,
        "selected_hypothesis_id": hypothesis["hypothesis_id"],
        "qualified_hypotheses": [hypothesis],
    })
    assert output["current_executable"] is False
    assert output["experiment_plan"] is None
    assert (
        "protocol:prediction_horizon_conflict:problem=80:hypothesis=40"
        in output["missing_capabilities"]
    )
