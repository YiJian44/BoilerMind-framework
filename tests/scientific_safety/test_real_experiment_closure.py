from datetime import datetime, timezone

import pytest

from boilermind.audit.execution_trace import (
    ExperimentExecutionTrace,
)

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
)

from boilermind.core.enums import (
    ExperimentStatus,
    ScientificVerdict,
)

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.experiment.real_runner_adapter import (
    RealSklearnExperimentRunner,
)

from boilermind.orchestration.real_experiment_loop import (
    execute_real_experiment,
)


def make_contract(
    *,
    problem_id="RP-XXX",
    hypothesis_id="H002",
    plan_id="PLAN-H002",
):
    return ExperimentContract(
        experiment_id=f"EXP-{hypothesis_id}",
        problem_id=problem_id,
        hypothesis_id=hypothesis_id,
        plan_id=plan_id,
        experiment_type=(
            "reference_model_comparison"
        ),
        candidate_models=[
            "ridge",
            "bayesianridge",
            "hgb",
        ],
        reference_models=["persistence"],
        primary_metric="MAE",
        secondary_metrics=["RMSE"],
        prediction_horizon_steps=40,
        sampling_interval_seconds=15,
        locked_test_used_for_selection=False,
        dataset_id="BOILER-REAL",
        dataset_hash="real-hash",
        input_variables=["feature_01"],
        target_variable="main_steam_mass_flow",
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        baseline_models=["persistence"],
        metrics=["MAE", "RMSE"],
        confirmation_criteria=[
            "all_candidates_worse_than_reference_on:"
            "MAE,RMSE",
        ],
        falsification_criteria=[
            "any_candidate_better_than_reference_on:"
            "MAE,RMSE",
        ],
        random_seed=42,
    )


def make_payload(
    *,
    candidate_metrics,
    reference_metrics,
):
    return {
        "experiment_id": "EXP-H002",
        "status": "completed",
        "dataset": {
            "path": "dataset.csv",
            "sha256": "a" * 64,
        },
        "reference_model": {
            "model_id": "persistence",
            "locked_test_metrics": (
                reference_metrics
            ),
        },
        "selected_model_by_validation": "ridge",
        "models": {
            model_id: {
                "fit_success": True,
                "fit_converged": True,
                "warnings": [],
                "failure_reason": None,
                "model_config": {},
                "validation_metrics": {},
                "locked_test_metrics": metrics,
                "train_samples": 100,
                "validation_samples": 20,
                "test_samples": 20,
                "random_seed": 42,
                "dataset_sha256": "a" * 64,
                "model_artifact": (
                    f"{model_id}.joblib"
                ),
                "prediction_artifact": (
                    f"{model_id}_pred.csv"
                ),
                "artifact_paths": [
                    f"{model_id}.joblib",
                    f"{model_id}_pred.csv",
                ],
            }
            for model_id, metrics in (
                candidate_metrics.items()
            )
        },
        "split": {
            "locked_test_used_for_selection": False,
        },
        "result_artifact": (
            "experiment_result.json"
        ),
        "completed_at": (
            "2026-08-19T12:00:00+08:00"
        ),
    }


def metric_set(
    mae,
    rmse,
):
    return {
        "mae_t_h": mae,
        "rmse_t_h": rmse,
        "r2": 0.5,
        "mbe_t_h": 0.0,
    }


class StubBackend:
    __test__ = False

    def __init__(self, payload):
        self.payload = payload

    def run(self, contract):
        return self.payload


def run_closure(
    payload,
    *,
    contract=None,
    tmp_path,
):
    runner = RealSklearnExperimentRunner(
        registry=ExperimentCapabilityRegistry(),
        backend=StubBackend(payload),
        output_dir=tmp_path,
    )

    return execute_real_experiment(
        contract or make_contract(),
        runner=runner,
    )


def test_supported_when_all_candidates_worse_than_reference(
    tmp_path,
):
    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    outcome = run_closure(
        payload,
        tmp_path=tmp_path,
    )

    assert outcome["audit"].execution_valid is True
    assert (
        outcome["criterion_assessment"]
        .confirmation_met
        is True
    )
    assert (
        outcome["criterion_assessment"]
        .falsification_met
        is False
    )
    assert (
        outcome["scientific_result"].verdict
        == ScientificVerdict.SUPPORTED
    )


def test_falsified_when_any_candidate_better_on_both(
    tmp_path,
):
    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.24, 0.34),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    outcome = run_closure(
        payload,
        tmp_path=tmp_path,
    )

    assert (
        outcome["criterion_assessment"]
        .confirmation_met
        is False
    )
    assert (
        outcome["criterion_assessment"]
        .falsification_met
        is True
    )
    assert (
        outcome["scientific_result"].verdict
        == ScientificVerdict.FALSIFIED
    )


def test_mixed_result_is_insufficient_evidence(
    tmp_path,
):
    """
    ridge better on MAE only (not RMSE), hgb better on
    RMSE only (not MAE): no candidate better on BOTH,
    and not all candidates worse on both.
    """

    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.20, 0.40),
            "bayesianridge": metric_set(
                0.26,
                0.34,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    outcome = run_closure(
        payload,
        tmp_path=tmp_path,
    )

    assert (
        outcome["criterion_assessment"]
        .confirmation_met
        is False
    )
    assert (
        outcome["criterion_assessment"]
        .falsification_met
        is False
    )
    assert (
        outcome["scientific_result"].verdict
        == ScientificVerdict.INSUFFICIENT_EVIDENCE
    )


def test_audit_failure_is_fail_closed(tmp_path):
    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    contract = make_contract()

    runner = RealSklearnExperimentRunner(
        registry=ExperimentCapabilityRegistry(),
        backend=StubBackend(payload),
        output_dir=tmp_path,
    )

    result, _trace = runner.run(contract)

    bad_trace = ExperimentExecutionTrace(
        experiment_id=contract.experiment_id,
        dataset_frozen=True,
        leakage_check_passed=False,
        baseline_valid=True,
        metric_check_passed=True,
        notes=["TEST-ONLY leakage failure"],
    )

    class FailingTraceRunner:
        is_test_only = True

        def run(self, _contract):
            return result, bad_trace

    outcome = execute_real_experiment(
        contract,
        runner=FailingTraceRunner(),
    )

    assert outcome["closure_ok"] is False
    assert (
        outcome["audit"].execution_valid
        is False
    )
    assert (
        "data_leakage_check_failed"
        in outcome["audit"].issues
    )
    assert (
        outcome["scientific_result"].verdict
        == ScientificVerdict.INSUFFICIENT_EVIDENCE
    )


def test_ids_propagate_through_result(tmp_path):
    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    contract = make_contract(
        problem_id="RP-XXX",
        hypothesis_id="H002",
        plan_id="PLAN-H002",
    )

    outcome = run_closure(
        payload,
        contract=contract,
        tmp_path=tmp_path,
    )

    result = outcome["experiment_result"]

    assert result.problem_id == "RP-XXX"
    assert result.hypothesis_id == "H002"
    assert result.plan_id == "PLAN-H002"
    assert result.experiment_id == "EXP-H002"


def test_candidate_locked_test_metrics_are_per_model(
    tmp_path,
):
    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    outcome = run_closure(
        payload,
        tmp_path=tmp_path,
    )

    candidates = (
        outcome["experiment_result"]
        .candidate_locked_test_metrics
    )

    assert set(candidates) == {
        "ridge",
        "bayesianridge",
        "hgb",
        "persistence",
    }

    for model_id, metrics in candidates.items():
        assert "MAE" in metrics
        assert "RMSE" in metrics


def test_unknown_criterion_format_fails_closed(tmp_path):
    contract = make_contract().model_copy(
        update={
            "confirmation_criteria": [
                "some_unknown_criterion",
            ],
        }
    )

    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    with pytest.raises(
        ValueError,
        match="unsupported_confirmation_criterion",
    ):
        run_closure(
            payload,
            contract=contract,
            tmp_path=tmp_path,
        )
def test_pairwise_model_relation_criteria_use_named_reference(tmp_path):
    contract = make_contract().model_copy(update={
        "confirmation_criteria": [
            "all_models_not_better_than_model_on:bayesianridge|ridge|MAE,RMSE"
        ],
        "falsification_criteria": [
            "any_model_better_than_model_on:bayesianridge|ridge|MAE,RMSE"
        ],
    })
    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(0.31, 0.41),
            "hgb": metric_set(0.32, 0.42),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )
    outcome = run_closure(payload, contract=contract, tmp_path=tmp_path)
    assessment = outcome["criterion_assessment"]
    assert assessment.confirmation_met is True
    assert assessment.falsification_met is False


def test_missing_candidate_metrics_fails_closed(tmp_path):
    contract = make_contract()

    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            # hgb missing on purpose
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    with pytest.raises(
        ValueError,
        match="executed_models_mismatch_contract",
    ):
        run_closure(
            payload,
            contract=contract,
            tmp_path=tmp_path,
        )
def test_extra_executed_model_fails_closed(tmp_path):
    contract = make_contract()

    payload = make_payload(
        candidate_metrics={
            "ridge": metric_set(0.30, 0.40),
            "bayesianridge": metric_set(
                0.31,
                0.41,
            ),
            "hgb": metric_set(0.32, 0.42),
            # model NOT in the contract:
            "lstm": metric_set(0.20, 0.30),
        },
        reference_metrics=metric_set(0.25, 0.35),
    )

    with pytest.raises(
        ValueError,
        match="executed_models_mismatch_contract",
    ):
        run_closure(
            payload,
            contract=contract,
            tmp_path=tmp_path,
        )
