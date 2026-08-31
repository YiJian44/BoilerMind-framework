from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
    ModelExperimentRecord,
    ScientificResult,
)
from boilermind.core.enums import ExperimentStatus, ScientificVerdict
from boilermind.reflection import (
    ExperimentOptimizationSuggestion,
    WhitelistedConfiguration,
    build_next_experiment_contract,
    optimization_contract_issues,
    optimize_experiment_parameters,
)


def _contract(model="transformer"):
    return ExperimentContract(
        experiment_id="EXP-1",
        hypothesis_id="H-1",
        plan_id="PLAN-1",
        dataset_id="DATA-1",
        dataset_hash="a" * 64,
        input_variables=["signal"],
        target_variable="target",
        train_split="chronological_train",
        validation_split="chronological_validation",
        test_split="locked_test",
        baseline_models=["reference"],
        candidate_models=[model],
        metrics=["MAE", "RMSE"],
        confirmation_criteria=["candidate_better_than_reference"],
        falsification_criteria=["candidate_not_better_than_reference"],
        window_steps=20,
        prediction_horizon_steps=40,
        sampling_interval_seconds=15,
        locked_test_used_for_selection=False,
    )


def _result(candidate_mae, baseline_mae, model="transformer"):
    now = datetime.now(timezone.utc)
    configuration = (
        {"d_model": 64, "learning_rate": 0.001}
        if model == "transformer"
        else {"hidden_size": 64, "learning_rate": 0.001}
    )
    return ExperimentResult(
        experiment_id="EXP-1",
        hypothesis_id="H-1",
        plan_id="PLAN-1",
        status=ExperimentStatus.COMPLETED,
        metrics={"MAE": candidate_mae},
        normalized_metrics={"MAE": candidate_mae},
        baseline_metrics={"MAE": baseline_mae},
        candidate_locked_test_metrics={model: {"MAE": candidate_mae}},
        model_records={model: ModelExperimentRecord(
            model_name=model,
            fit_success=True,
            fit_converged=True,
            model_configuration=configuration,
        )},
        started_at=now,
        completed_at=now,
    )


def _scientific(verdict):
    return ScientificResult(
        hypothesis_id="H-1",
        experiment_id="EXP-1",
        verdict=verdict,
        rationale="existing scientific result",
    )


def test_high_error_generates_whitelisted_optimization_suggestion():
    suggestion = optimize_experiment_parameters(
        _contract(),
        _result(2.0, 1.0),
        _scientific(ScientificVerdict.INSUFFICIENT_EVIDENCE),
    )
    assert suggestion.performance_analysis.status == "high_error_relative_to_baseline"
    assert suggestion.next_configuration.window_size == 40
    assert suggestion.next_configuration.d_model == 64
    assert suggestion.next_configuration.learning_rate == 0.001
    assert suggestion.changed_parameters == ["window_size"]


def test_parameter_whitelist_and_model_binding_are_enforced():
    with pytest.raises(ValidationError):
        WhitelistedConfiguration(model_name="transformer", d_model=96)
    with pytest.raises(ValidationError):
        WhitelistedConfiguration(model_name="lstm", d_model=64)
    with pytest.raises(ValidationError):
        WhitelistedConfiguration(model_name="gru", learning_rate=0.002)


def test_stable_performance_keeps_configuration():
    suggestion = optimize_experiment_parameters(
        _contract("lstm"),
        _result(0.5, 1.0, "lstm"),
        _scientific(ScientificVerdict.SUPPORTED),
    )
    assert suggestion.performance_analysis.status == "stable"
    assert suggestion.changed_parameters == []
    assert suggestion.next_configuration.window_size == 20
    assert suggestion.next_configuration.hidden_size == 64


def test_suggestion_builds_next_contract_without_changing_locked_test_semantics():
    contract = _contract()
    suggestion = optimize_experiment_parameters(
        contract,
        _result(2.0, 1.0),
        _scientific(ScientificVerdict.INSUFFICIENT_EVIDENCE),
    )
    next_contract = build_next_experiment_contract(
        contract,
        suggestion,
        next_experiment_id="EXP-2",
        next_plan_id="PLAN-2",
    )
    assert next_contract.experiment_id == "EXP-2"
    assert next_contract.window_steps == 40
    assert next_contract.prediction_horizon_steps == 40
    assert next_contract.test_split == contract.test_split
    assert next_contract.locked_test_used_for_selection is False
    assert next_contract.metrics == contract.metrics
    assert next_contract.optimization_suggestion == suggestion.model_dump(mode="json")
    assert contract.optimization_suggestion is None


def test_suggestion_schema_can_round_trip():
    suggestion = optimize_experiment_parameters(
        _contract(),
        _result(2.0, 1.0),
        _scientific(ScientificVerdict.FALSIFIED),
    )
    assert ExperimentOptimizationSuggestion.model_validate_json(
        suggestion.model_dump_json()
    ) == suggestion


class _Capability:
    def __init__(self, window_steps):
        self.window_steps = window_steps

    @staticmethod
    def prediction_horizon_steps_value():
        return 40

    @staticmethod
    def sampling_interval_seconds_value():
        return 15

    @staticmethod
    def check_executable(**_requirements):
        return SimpleNamespace(missing_capabilities=[])


class _ModelRegistry:
    @staticmethod
    def compatible_with_capability(_capability, **_requirements):
        return [SimpleNamespace(model_name="transformer")]

    @staticmethod
    def get(_model_name):
        return SimpleNamespace(required_features=1)


def test_legal_suggestion_passes_existing_capability_checks():
    contract = _contract()
    suggestion = optimize_experiment_parameters(
        contract,
        _result(2.0, 1.0),
        _scientific(ScientificVerdict.INSUFFICIENT_EVIDENCE),
    )
    next_contract = build_next_experiment_contract(
        contract,
        suggestion,
        next_experiment_id="EXP-2",
        next_plan_id="PLAN-2",
    )
    assert optimization_contract_issues(
        next_contract,
        capability=_Capability(window_steps=40),
        model_registry=_ModelRegistry(),
    ) == []


def test_unsupported_optimization_does_not_modify_current_contract():
    contract = _contract()
    before = contract.model_dump(mode="json")
    suggestion = optimize_experiment_parameters(
        contract,
        _result(2.0, 1.0),
        _scientific(ScientificVerdict.INSUFFICIENT_EVIDENCE),
    )
    proposed = build_next_experiment_contract(
        contract,
        suggestion,
        next_experiment_id="EXP-2",
        next_plan_id="PLAN-2",
    )
    issues = optimization_contract_issues(
        proposed,
        capability=_Capability(window_steps=20),
        model_registry=_ModelRegistry(),
    )
    assert "unsupported_window_size:40" in issues
    assert contract.model_dump(mode="json") == before
    assert contract.optimization_suggestion is None
