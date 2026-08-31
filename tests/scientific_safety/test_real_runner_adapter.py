from datetime import datetime, timezone

import pytest

from boilermind.core.contracts import (
    ExperimentContract,
)

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.experiment.real_runner_adapter import (
    RealSklearnExperimentRunner,
)
from boilermind.models.execution_environment import ExecutionEnvironment


def make_contract(
    *,
    hypothesis_id="H002",
    candidate_models=None,
    metrics=None,
):
    return ExperimentContract(
        experiment_id=f"EXP-{hypothesis_id}",
        hypothesis_id=hypothesis_id,
        plan_id=f"PLAN-{hypothesis_id}",
        dataset_id="BOILER-REAL",
        dataset_hash="real-hash",
        input_variables=["feature_01"],
        target_variable="main_steam_mass_flow",
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        baseline_models=["persistence"],
        candidate_models=(
            candidate_models
            or [
                "ridge",
                "bayesianridge",
                "hgb",
            ]
        ),
        metrics=(
            metrics
            or [
                "MAE",
                "RMSE",
                "R2",
                "MBE",
            ]
        ),
        confirmation_criteria=[
            "all_candidates_worse_than_persistence"
        ],
        falsification_criteria=[
            "any_candidate_better_than_persistence"
        ],
        random_seed=42,
    )


def make_payload(
    *,
    experiment_id="EXP-H002",
):
    return {
        "schema_version": (
            "boilermind.real_sklearn_experiment.v1"
        ),
        "experiment_id": experiment_id,
        "status": "completed",
        "dataset": {
            "path": "dataset.csv",
            "sha256": "a" * 64,
        },
        "reference_model": {
            "model_id": "persistence",
            "locked_test_metrics": {
                "mae_t_h": 0.30,
                "rmse_t_h": 0.40,
                "r2": 0.80,
                "mbe_t_h": 0.01,
            },
        },
        "selected_model_by_validation": "ridge",
        "models": {
            "ridge": {
                "fit_success": True,
                "fit_converged": True,
                "warnings": [],
                "failure_reason": None,
                "model_config": {"alpha": 1.0},
                "validation_metrics": {
                    "mae_t_h": 0.20,
                },
                "locked_test_metrics": {
                    "mae_t_h": 0.25,
                    "rmse_t_h": 0.35,
                    "r2": 0.85,
                    "mbe_t_h": 0.00,
                },
                "train_samples": 100,
                "validation_samples": 20,
                "test_samples": 20,
                "random_seed": 42,
                "dataset_sha256": "a" * 64,
                "model_artifact": "ridge.joblib",
                "prediction_artifact": (
                    "ridge_pred.csv"
                ),
                "artifact_paths": [
                    "ridge.joblib",
                    "ridge_pred.csv",
                ],
            },
            "bayesianridge": {
                "fit_success": True,
                "fit_converged": True,
                "warnings": [],
                "failure_reason": None,
                "model_config": {},
                "validation_metrics": {},
                "locked_test_metrics": {
                    "mae_t_h": 0.26,
                    "rmse_t_h": 0.36,
                    "r2": 0.84,
                    "mbe_t_h": 0.01,
                },
                "train_samples": 100,
                "validation_samples": 20,
                "test_samples": 20,
                "random_seed": 42,
                "dataset_sha256": "a" * 64,
                "model_artifact": (
                    "bayesianridge.joblib"
                ),
                "prediction_artifact": (
                    "bayesianridge_pred.csv"
                ),
                "artifact_paths": [
                    "bayesianridge.joblib",
                    "bayesianridge_pred.csv",
                ],
            },
            "hgb": {
                "fit_success": True,
                "fit_converged": True,
                "warnings": [],
                "failure_reason": None,
                "model_config": {},
                "validation_metrics": {},
                "locked_test_metrics": {
                    "mae_t_h": 0.22,
                    "rmse_t_h": 0.32,
                    "r2": 0.88,
                    "mbe_t_h": -0.01,
                },
                "train_samples": 100,
                "validation_samples": 20,
                "test_samples": 20,
                "random_seed": 42,
                "dataset_sha256": "a" * 64,
                "model_artifact": "hgb.joblib",
                "prediction_artifact": (
                    "hgb_pred.csv"
                ),
                "artifact_paths": [
                    "hgb.joblib",
                    "hgb_pred.csv",
                ],
            },
        },
        "split": {
            "locked_test_used_for_selection": False,
        },
        "result_artifact": "experiment_result.json",
        "completed_at": "2026-08-19T12:00:00+08:00",
    }


class StubBackend:
    """
    TEST-ONLY stub for adapter mapping logic.

    It never enters the scientific chain; it only
    simulates the real backend payload shape.
    """

    __test__ = False

    def __init__(self, payload):
        self.payload = payload

    def run(self, contract):
        return self.payload


def make_runner(
    payload=None,
    *,
    output_dir,
):
    return RealSklearnExperimentRunner(
        registry=ExperimentCapabilityRegistry(
            environment=ExecutionEnvironment(
                os="test", python_version="3.11.9",
                sklearn_available=True, torch_available=False,
                cuda_available=False, gpu_available=False,
            )
        ),
        backend=StubBackend(
            payload or make_payload()
        ),
        output_dir=output_dir,
    )


def test_runner_requires_experiment_contract():
    runner = make_runner(output_dir=".")

    with pytest.raises(TypeError):
        runner.run({"not": "a contract"})


def test_runner_rejects_non_enabled_candidate_model(
    tmp_path,
):
    runner = make_runner(output_dir=tmp_path)

    contract = make_contract(
        candidate_models=[
            "ridge",
            "transformer",
        ],
    )

    with pytest.raises(
        ValueError,
        match="candidate_model_not_enabled",
    ):
        runner.run(contract)


def test_runner_rejects_unsupported_metric(tmp_path):
    runner = make_runner(output_dir=tmp_path)

    contract = make_contract(
        metrics=[
            "MAE",
            "MAPE",
        ],
    )

    with pytest.raises(
        ValueError,
        match="unsupported_metrics",
    ):
        runner.run(contract)


def test_runner_maps_payload_to_result_and_trace(
    tmp_path,
):
    runner = make_runner(output_dir=tmp_path)

    contract = make_contract()

    result, trace = runner.run(contract)

    assert result.experiment_id == "EXP-H002"
    assert result.hypothesis_id == "H002"

    assert set(
        result.candidate_locked_test_metrics
    ) == {
        "ridge",
        "bayesianridge",
        "hgb",
        "persistence",
    }

    assert (
        result.candidate_locked_test_metrics[
            "hgb"
        ]["MAE"]
        == 0.22
    )

    assert (
        result.candidate_locked_test_metrics[
            "persistence"
        ]["RMSE"]
        == 0.40
    )

    # Representative flat metrics follow the
    # validation-selected model vs persistence.
    assert result.metrics["MAE"] == 0.25
    assert result.baseline_metrics["MAE"] == 0.30

    assert (
        "REAL_SKLEARN_EXECUTION"
        in result.execution_notes
    )

    assert (
        "locked_test_not_used_for_selection"
        in result.execution_notes
    )

    assert set(result.model_records) == {
        "ridge",
        "bayesianridge",
        "hgb",
    }

    ridge_record = result.model_records["ridge"]

    assert ridge_record.fit_success is True
    assert ridge_record.fit_converged is True
    assert ridge_record.model_configuration == {
        "alpha": 1.0
    }
    assert ridge_record.train_samples == 100
    assert ridge_record.dataset_sha256 == "a" * 64

    assert trace.experiment_id == "EXP-H002"
    assert trace.dataset_frozen is True
    assert trace.leakage_check_passed is True
    assert trace.baseline_valid is True
    assert trace.metric_check_passed is True


def test_runner_fails_closed_when_locked_test_used_for_selection(
    tmp_path,
):
    payload = make_payload()

    payload["split"] = {
        "locked_test_used_for_selection": True,
    }

    runner = make_runner(
        payload,
        output_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="locked_test_used_for_selection",
    ):
        runner.run(make_contract())


def test_runner_fails_closed_on_experiment_id_mismatch(
    tmp_path,
):
    runner = make_runner(
        make_payload(
            experiment_id="EXP-OTHER",
        ),
        output_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="experiment_id_mismatch",
    ):
        runner.run(make_contract())


def test_contract_keeps_hypothesis_id_chain(tmp_path):
    contract = make_contract(
        hypothesis_id="H002",
    )

    runner = make_runner(output_dir=tmp_path)

    result, trace = runner.run(contract)

    assert result.hypothesis_id == "H002"
    assert trace.experiment_id == "EXP-H002"
    assert result.artifacts

    assert (
        datetime.now(timezone.utc) - result.started_at
    ).total_seconds() < 60
    assert result.completed_at is not None
