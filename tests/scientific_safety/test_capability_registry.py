import os

import pytest

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)
from boilermind.models.execution_environment import ExecutionEnvironment


def make_registry():
    return ExperimentCapabilityRegistry(
        environment=ExecutionEnvironment(
            os="test",
            python_version="3.11.9",
            sklearn_available=True,
            torch_available=False,
            cuda_available=False,
            gpu_available=False,
        )
    )


def test_answers_core_capability_questions():
    registry = make_registry()

    assert set(registry.available_models()) == {
        "ridge",
        "bayesianridge",
        "hgb",
        "svr",
        "rf",
        "mlp",
        "elasticnet",
        "pls",
        "knn",
        "gpr",
    }

    assert (
        registry.reference_model_id()
        == "persistence"
    )

    assert set(registry.metrics()) == {
        "MAE",
        "RMSE",
        "R2",
        "MBE",
    }

    assert set(registry.operations()) == {
        "model_comparison",
        "reference_model_comparison",
        "chronological_validation",
        "locked_test_evaluation",
        "regime_stratified_evaluation",
    }

    assert (
        registry.supports_feature_intervention()
        is False
    )

    assert registry.supports_locked_test() is True

    assert (
        registry.sampling_interval_seconds_value()
        == 15
    )

    assert (
        registry.prediction_horizon_steps_value()
        == 40
    )


def test_executable_models_are_derived_from_model_registry():
    """
    No hard-coded model list: the registry must derive
    executability from the ModelRegistry catalog.
    """

    registry = make_registry()

    available = set(registry.available_models())

    assert "gpr" in available
    assert "transformer" not in available
    assert "lstm" not in available
    assert "tcn" not in available
    assert "mtgnn" not in available
    assert "csdi" not in available
    assert "psfa_v0" not in available
    assert "candidate_xgboost_direct_volume_20m" not in (
        available
    )

    assert "ridge" in available
    assert "bayesianridge" in available
    assert "hgb" in available


def test_unavailable_models_come_with_deterministic_reasons():
    registry = make_registry()

    reasons = registry.unavailable_models_with_reasons()

    assert "transformer" in reasons
    assert "torch_not_installed" in reasons["transformer"]

    assert "gpr" not in reasons

    assert "tcn" in reasons
    assert "torch_not_installed" in reasons["tcn"]
    assert "checkpoint_incompatible" not in reasons["tcn"]

    assert "candidate_xgboost_direct_volume_20m" in (
        reasons
    )
    assert "xgboost_not_installed" in (
        reasons[
            "candidate_xgboost_direct_volume_20m"
        ]
    )

    assert "psfa_v0" in reasons
    assert "checkpoint_incompatible" in reasons["psfa_v0"]


def test_reference_model_comparison_is_executable():
    registry = make_registry()

    match = registry.check_executable(
        required_operations=[
            "reference_model_comparison",
            "chronological_validation",
            "locked_test_evaluation",
        ],
        required_models=[
            "ridge",
            "bayesianridge",
            "hgb",
            "persistence",
        ],
        required_metrics=[
            "MAE",
            "RMSE",
        ],
    )

    assert match.executable is True
    assert match.missing_capabilities == []


def test_feature_intervention_fails_closed():
    registry = make_registry()

    match = registry.check_executable(
        required_operations=[
            "model_comparison",
        ],
        requires_feature_intervention=True,
    )

    assert match.executable is False
    assert (
        "operation:feature_intervention"
        in match.missing_capabilities
    )


def test_unknown_requirements_are_reported():
    registry = make_registry()

    match = registry.check_executable(
        required_operations=[
            "feature_intervention",
        ],
        required_models=[
            "lstm",
        ],
        required_metrics=[
            "MAPE",
        ],
    )

    assert match.executable is False
    assert (
        "operation:feature_intervention"
        in match.missing_capabilities
    )
    assert "model:lstm" in match.missing_capabilities
    assert "metric:mape" in match.missing_capabilities


def test_normalization_of_operation_names():
    registry = make_registry()

    match = registry.check_executable(
        required_operations=[
            "reference model comparison",
        ],
        required_metrics=[
            "mae",
        ],
    )

    assert match.executable is True
    assert (
        "reference_model_comparison"
        in match.matched_operations
    )


def test_scientific_context_shape():
    registry = make_registry()

    context = registry.to_scientific_context()

    assert context["enabled_experiment_models"] == [
        "bayesianridge",
        "elasticnet",
        "gpr",
        "hgb",
        "knn",
        "mlp",
        "pls",
        "rf",
        "ridge",
        "svr",
    ]

    assert (
        context["reference_model"]
        == "persistence"
    )

    assert set(
        context["supported_experiment_operations"]
    ) == {
        "model_comparison",
        "reference_model_comparison",
        "chronological_validation",
        "locked_test_evaluation",
        "regime_stratified_evaluation",
    }

    assert (
        context["feature_intervention_supported"]
        is False
    )

    assert (
        context["dataset_contract"][
            "real_industrial_data"
        ]
        is True
    )


def test_dataset_metadata_is_real_and_frozen():
    registry = make_registry()

    assert registry.dataset_exists() is True

    snapshot = registry.snapshot()

    assert len(snapshot["dataset"]["sha256"]) == 64
    assert (
        snapshot["dataset"]["real_industrial_data"]
        is True
    )
    assert snapshot["dataset"]["frozen"] is True
    assert (
        snapshot["dataset"]["leakage_policy_verified"]
        is True
    )
    assert (
        snapshot["splits"][
            "locked_test_used_for_selection"
        ]
        is False
    )


def test_variables_and_targets_are_available():
    registry = make_registry()

    variables = registry.available_variables()
    targets = registry.available_target_variables()

    assert len(variables) == 30
    assert variables[0] == "feature_01"
    assert targets == ["main_steam_mass_flow"]

    match = registry.check_executable(
        required_variables=[
            "feature_01",
            "main_steam_mass_flow",
        ],
    )

    assert match.executable is True


def test_default_dataset_path_is_project_internal():
    registry = make_registry()

    path = str(registry.dataset_path.resolve())

    assert path.endswith(
        os.path.join(
            "resources",
            "data",
            "shortperiod_new.csv",
        )
    )

    assert "_bm_sync_tmp" not in path
    assert registry.dataset_path.is_file()


def test_env_var_dataset_path_overrides_default(
    tmp_path,
    monkeypatch,
):
    custom = tmp_path / "custom.csv"
    custom.write_bytes(b"0,1,2\n3,4,5\n")

    monkeypatch.setenv(
        "BOILERMIND_REAL_DATASET_PATH",
        str(custom),
    )

    registry = ExperimentCapabilityRegistry()

    assert registry.dataset_path == custom.resolve()
    assert registry.dataset_hash() is not None


def test_missing_dataset_fails_closed_without_fallback(
    monkeypatch,
):
    missing = (
        r"D:\BoilerMind-Trusted\resources\data"
        r"\does_not_exist_shortperiod_new.csv"
    )

    monkeypatch.setenv(
        "BOILERMIND_REAL_DATASET_PATH",
        missing,
    )

    with pytest.raises(
        FileNotFoundError,
        match="dataset_path_not_found_no_fallback",
    ):
        ExperimentCapabilityRegistry()
