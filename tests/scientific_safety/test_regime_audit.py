from datetime import datetime, timezone

from boilermind.audit.execution_trace import ExperimentExecutionTrace
from boilermind.audit.experiment_auditor import audit_experiment
from boilermind.core.contracts import ExperimentContract, ExperimentResult
from boilermind.core.enums import ExperimentStatus


def _contract() -> ExperimentContract:
    return ExperimentContract(
        experiment_id="EXP-REGIME",
        problem_id="P",
        hypothesis_id="H",
        plan_id="PL",
        experiment_type="regime_stratified_evaluation",
        required_operations=["regime_stratified_evaluation"],
        dataset_id="D",
        dataset_hash="hash",
        input_variables=["feature_1"],
        target_variable="steam_volumetric_flow",
        train_split="train",
        validation_split="validation",
        test_split="locked_test",
        baseline_models=["persistence"],
        candidate_models=["ridge"],
        metrics=["MAE"],
        confirmation_criteria=["criterion"],
        falsification_criteria=["criterion"],
    )


def _trace() -> ExperimentExecutionTrace:
    return ExperimentExecutionTrace(
        experiment_id="EXP-REGIME",
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
    )


def _result(regime_metrics):
    now = datetime(2026, 8, 23, tzinfo=timezone.utc)
    return ExperimentResult(
        experiment_id="EXP-REGIME",
        problem_id="P",
        hypothesis_id="H",
        plan_id="PL",
        status=ExperimentStatus.COMPLETED,
        metrics={"MAE": 1.0},
        raw_metrics={"MAE": 1.0},
        normalized_metrics={"MAE": 1.0},
        regime_metrics=regime_metrics,
        started_at=now,
        completed_at=now,
    )


def test_regime_audit_accepts_complete_direction_metrics():
    metrics = {
        "ridge": {
            regime: {"MAE": 1.0, "RMSE": 1.2, "MBE": 0.1, "sample_count": 8.0}
            for regime in ("ramp_up", "ramp_down")
        }
    }
    audit = audit_experiment(_contract(), _result(metrics), _trace())
    assert audit.execution_valid is True
    assert audit.issues == []


def test_regime_audit_rejects_missing_direction_metrics():
    metrics = {
        "ridge": {
            "ramp_up": {"MAE": 1.0, "RMSE": 1.2, "MBE": 0.1, "sample_count": 8.0}
        }
    }
    audit = audit_experiment(_contract(), _result(metrics), _trace())
    assert audit.execution_valid is False
    assert audit.metric_check_passed is False
    assert "regime_missing:ridge:ramp_down" in audit.issues
