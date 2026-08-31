import numpy as np
import pytest

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.models import (
    ExecutionEnvironment,
    build_default_registry,
)


EXPECTED_MODELS = {
    "persistence",
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
    "transformer",
    "lstm",
    "dlinear",
    "gru",
    "patchtst",
    "itransformer",
    "timesnet",
    "mtgnn",
    "csdi",
    "tcn",
    "psfa_v0",
    "psfa_v20",
    "psfa_v80",
    "legacy_tcn_1min",
    "legacy_tcn_2min",
    "legacy_transformer_5min",
    "legacy_transformer_10min",
    "candidate_xgboost_direct_volume_20m",
    "candidate_lstm_direct_volume_20m",
}


def make_registry():
    return build_default_registry()


def names(specs):
    return [spec.model_name for spec in specs]


def test_default_catalog_registers_all_audited_models():
    registry = make_registry()

    assert set(registry.names()) == EXPECTED_MODELS


def test_every_spec_has_required_registry_fields():
    registry = make_registry()

    required_fields = [
        "model_name",
        "framework",
        "supported_tasks",
        "required_input_type",
        "required_features",
        "sequence_required",
        "window_requirements",
        "horizon_capability",
        "supported_targets",
        "training_available",
        "inference_available",
        "checkpoint_available",
        "checkpoint_compatibility",
        "supported_metrics",
        "supports_uncertainty",
        "compute_cost",
        "status",
    ]

    for spec in registry.list_models():
        for field in required_fields:
            value = getattr(spec, field)
            assert value is not None, (
                f"{spec.model_name}.{field} missing"
            )

        compatibility = (
            spec.checkpoint_compatibility
        )

        assert (
            "compatible"
            in compatibility
        )
        assert (
            "mismatches"
            in compatibility
        )


def test_current_contract_filter_excludes_incompatible_legacy():
    registry = make_registry()

    suitable = registry.filter_compatible(
        features=30,
        window_steps=20,
        horizon_steps=40,
        sampling_interval_seconds=15,
        target="main_steam_mass_flow",
        checkpoint_compatible_required=True,
    )

    suitable_names = set(names(suitable))

    assert "persistence" in suitable_names
    assert "ridge" in suitable_names
    assert "bayesianridge" in suitable_names
    assert "hgb" in suitable_names
    assert "transformer" in suitable_names
    assert "lstm" in suitable_names

    for incompatible in [
        "mtgnn",
        "csdi",
        "tcn",
        "psfa_v0",
        "psfa_v20",
        "psfa_v80",
        "legacy_tcn_1min",
        "legacy_tcn_2min",
        "legacy_transformer_5min",
        "legacy_transformer_10min",
        "candidate_xgboost_direct_volume_20m",
        "candidate_lstm_direct_volume_20m",
    ]:
        assert incompatible not in suitable_names


def test_training_required_excludes_deep_and_legacy():
    registry = make_registry()

    trainable = registry.filter_compatible(
        training_required=True,
    )

    trainable_names = set(names(trainable))

    assert trainable_names == {
        "persistence",
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
        "transformer",
        "lstm",
        "gru",
        "dlinear",
    }


def test_volume_task_models_are_discoverable():
    registry = make_registry()

    volume_models = registry.filter_compatible(
        tasks=["steam_volume_forecast"],
        target="steam_volumetric_flow",
    )

    volume_names = set(names(volume_models))

    assert "psfa_v0" in volume_names
    assert "psfa_v80" in volume_names
    assert (
        "candidate_xgboost_direct_volume_20m"
        in volume_names
    )


def test_compatible_with_capability_returns_executable_pool():
    registry = make_registry()
    capability = ExperimentCapabilityRegistry(
        environment=ExecutionEnvironment(
            os="test", python_version="3.11.9",
            sklearn_available=True, torch_available=False,
            cuda_available=False, gpu_available=False,
        )
    )

    pool = registry.compatible_with_capability(
        capability,
        target="main_steam_mass_flow",
        metrics=["MAE", "RMSE"],
    )

    assert names(pool) == [
        "bayesianridge",
        "elasticnet",
        "gpr",
        "hgb",
        "knn",
        "mlp",
        "persistence",
        "pls",
        "rf",
        "ridge",
        "svr",
    ]


def test_unknown_model_raises():
    registry = make_registry()

    with pytest.raises(KeyError):
        registry.get("lstm_unknown")

    with pytest.raises(KeyError):
        registry.build_adapter("lstm_unknown")


def test_sklearn_adapter_unified_interface():
    registry = make_registry()

    adapter = registry.build_adapter("ridge")

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 600))
    y = X[:, 0] * 2.0 + rng.normal(
        scale=0.1,
        size=120,
    )

    adapter.fit(X, y)

    predictions = adapter.predict(X)

    assert predictions.shape == (120,)
    assert np.all(predictions >= 0.0)

    metrics = adapter.evaluate(y, predictions)

    assert set(metrics) == {
        "mae_t_h",
        "rmse_t_h",
        "r2",
        "mbe_t_h",
    }


def test_persistence_adapter_uses_source_values():
    registry = make_registry()

    adapter = registry.build_adapter("persistence")

    X = np.zeros((10, 30))
    source = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )

    adapter.fit(X, source)

    np.testing.assert_allclose(
        adapter.predict(X),
        source,
    )


def test_legacy_adapter_refuses_incompatible_checkpoint():
    registry = make_registry()

    with pytest.raises(
        ValueError,
        match="checkpoint_incompatible_reuse_refused",
    ):
        registry.build_adapter("tcn", reuse_checkpoint=True)


def test_deep_adapter_fit_fails_closed_without_torch(monkeypatch):
    registry = make_registry()

    adapter = registry.build_adapter("transformer")
    monkeypatch.setattr(
        adapter,
        "_load_torch",
        lambda: (_ for _ in ()).throw(
            RuntimeError("torch_required:torch_not_installed:transformer")
        ),
    )

    with pytest.raises(RuntimeError, match="torch_required"):
        adapter.fit(
            np.zeros((30, 20, 30)),
            np.zeros(30),
        )


def test_legacy_deep_adapter_requires_torch_for_predict():
    registry = make_registry()

    adapter = registry.build_adapter("transformer")

    with pytest.raises(
        RuntimeError,
        match="model_not_fitted",
    ):
        adapter.predict(
            np.zeros((30, 30))
        )


def test_checkpoint_compatibility_reasons_are_explicit():
    registry = make_registry()

    tcn = registry.get("tcn")
    mtgnn = registry.get("mtgnn")

    assert tcn.checkpoint_compatibility["compatible"] is False
    assert "window_steps" in tcn.checkpoint_compatibility[
        "mismatches"
    ]

    assert (
        mtgnn.checkpoint_compatibility["compatible"]
        is False
    )
    assert "prediction_horizon" in (
        mtgnn.checkpoint_compatibility["mismatches"]
    )


def test_uncertainty_flags():
    registry = make_registry()

    assert registry.get("bayesianridge").supports_uncertainty
    assert registry.get("gpr").supports_uncertainty
    assert registry.get("csdi").supports_uncertainty
    assert not registry.get("ridge").supports_uncertainty
