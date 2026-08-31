from datetime import datetime, timezone

import numpy as np
import pytest

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
)

from boilermind.core.enums import ExperimentStatus

from boilermind.experiment.backends import (
    CUDABackend,
    ExecutionBackend,
    LocalCPUBackend,
    RemoteCPUBackend,
    get_execution_backend,
)

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.experiment.real_runner_adapter import (
    RealSklearnExperimentRunner,
)

from boilermind.models import (
    BaseModelAdapter,
    ExecutionEnvironment,
    ModelRegistry,
    ModelSpec,
    PersistenceModelAdapter,
    SklearnModelAdapter,
    TorchModelAdapter,
    build_default_registry,
    model_status_matrix,
)


def make_fake_env(
    *,
    torch_available=False,
    cuda_available=False,
):
    return ExecutionEnvironment(
        os="Linux",
        python_version="3.11.9",
        sklearn_available=True,
        torch_available=torch_available,
        cuda_available=cuda_available,
        gpu_available=cuda_available,
        gpu_device=(
            "TestGPU"
            if cuda_available
            else None
        ),
        optional_dependencies={
            "dashscope": False,
            "xgboost": False,
            "pandas": True,
            "numpy": True,
            "joblib": True,
            "scipy": True,
            "openai": True,
        },
        metadata={
            "cpu_count": 8,
            "memory_gb": 16.0,
        },
    )


def test_execution_environment_detects_capabilities():
    env = ExecutionEnvironment.detect()

    payload = env.to_dict()

    assert payload["os"]
    assert payload["python_version"].startswith("3.")
    assert payload["sklearn_available"] is True
    assert "torch_available" in payload
    assert "cuda_available" in payload
    assert "gpu_available" in payload
    assert "optional_dependencies" in payload
    assert "metadata" in payload


def test_torch_missing_fails_closed():
    capability = ExperimentCapabilityRegistry(
        environment=make_fake_env(
            torch_available=False,
        )
    )

    assert "transformer" not in (
        capability.available_models()
    )

    reasons = capability.unavailable_models_with_reasons()

    assert "torch_required:torch_not_installed" in (
        reasons["transformer"]
    )

    # Capability discovery must use the injected environment rather than the
    # packages installed in the interpreter running the test.


def test_cuda_required_but_unavailable():
    spec = ModelSpec(
        model_name="cuda_only_model",
        framework="torch",
        supported_tasks=["mass_flow_forecast"],
        required_input_type="sequence_window",
        required_features=30,
        sequence_required=True,
        window_requirements={
            "steps": 20,
            "sampling_interval_seconds": 15,
        },
        horizon_capability={
            "steps": 40,
            "supported_steps": [40],
        },
        supported_targets=["main_steam_mass_flow"],
        training_available=True,
        trainable=True,
        inference_available=True,
        inference_supported=True,
        checkpoint_available=True,
        checkpoint_required=True,
        checkpoint_compatibility={
            "compatible": True,
            "mismatches": [],
        },
        supported_metrics=["MAE"],
        supports_uncertainty=False,
        compute_cost="minutes",
        status="checkpoint_ready",
        source_code_path="vendor/legacy/lstm.py",
        requires_torch=True,
        requires_cuda=True,
        cpu_supported=False,
        gpu_supported=True,
    )

    registry = ModelRegistry([spec])

    capability = ExperimentCapabilityRegistry(
        environment=make_fake_env(
            torch_available=True,
            cuda_available=False,
        )
    )

    executable = capability._discover_executable_models(
        make_fake_env(
            torch_available=True,
            cuda_available=False,
        ),
        catalog=registry,
    )

    assert "cuda_only_model" not in executable

    backend = CUDABackend(
        environment=make_fake_env(
            torch_available=True,
            cuda_available=False,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="cuda_required_but_unavailable",
    ):
        backend.execute(None)


def test_sklearn_model_executable():
    capability = ExperimentCapabilityRegistry()

    assert "ridge" in capability.available_models()
    assert "elasticnet" in capability.available_models()

    registry = build_default_registry()

    adapter = registry.build_adapter("ridge")

    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 20))
    y = X[:, 0] * 2.0 + 1.0

    adapter.fit(X, y)

    predictions = adapter.predict(X)

    assert predictions.shape == (50,)

    metrics = adapter.evaluate(y, predictions)

    assert set(metrics) == {
        "mae_t_h",
        "rmse_t_h",
        "r2",
        "mbe_t_h",
    }


def test_adapter_interface_is_unified():
    assert issubclass(BaseModelAdapter.__class__, type)

    registry = build_default_registry()

    adapters = [
        registry.build_adapter("ridge"),
        registry.build_adapter("persistence"),
        registry.build_adapter("transformer"),
    ]

    for adapter in adapters:
        assert callable(adapter.fit)
        assert callable(adapter.predict)
        assert callable(adapter.evaluate)

    assert isinstance(
        registry.build_adapter("ridge"),
        SklearnModelAdapter,
    )
    assert isinstance(
        registry.build_adapter("persistence"),
        PersistenceModelAdapter,
    )
    assert isinstance(
        registry.build_adapter("transformer"),
        TorchModelAdapter,
    )


def test_registry_to_adapter_resolution():
    registry = build_default_registry()

    assert (
        registry.get("ridge").model_name
        == "ridge"
    )

    assert isinstance(
        registry.build_adapter("bayesianridge"),
        SklearnModelAdapter,
    )

    assert isinstance(
        registry.build_adapter("lstm"),
        TorchModelAdapter,
    )

    with pytest.raises(KeyError):
        registry.build_adapter("no_such_model")


def test_contract_to_runner_to_adapter_call(tmp_path):
    """
    ExperimentContract -> Runner -> ModelRegistry -> Adapter
    -> actual model code (real sklearn Ridge).
    """

    contract = ExperimentContract(
        experiment_id="EXP-ENV-CALL",
        problem_id="RP-ENV",
        hypothesis_id="H999",
        plan_id="PLAN-H999",
        experiment_type="reference_model_comparison",
        candidate_models=["ridge"],
        reference_models=["persistence"],
        primary_metric="MAE",
        metrics=["MAE", "RMSE"],
        confirmation_criteria=[
            "all_candidates_worse_than_reference_on:MAE"
        ],
        falsification_criteria=[
            "any_candidate_better_than_reference_on:MAE"
        ],
        dataset_id="BOILER-REAL",
        dataset_hash="real",
        input_variables=["feature_01"],
        target_variable="main_steam_mass_flow",
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        baseline_models=["persistence"],
        random_seed=42,
    )

    runner = RealSklearnExperimentRunner(output_dir=tmp_path)

    # Runner resolves the model through the registry.
    spec = runner.model_registry.get("ridge")

    assert spec.framework == "sklearn"

    adapter = runner.model_registry.build_adapter(
        "ridge"
    )

    assert isinstance(adapter, SklearnModelAdapter)

    result, trace = runner.execute(contract)

    assert "ridge" in (
        result.candidate_locked_test_metrics
    )
    assert trace.dataset_frozen is True
    assert "ridge" in result.model_records


def test_unsupported_model_fail_closed():
    contract = ExperimentContract(
        experiment_id="EXP-UNSUPPORTED",
        problem_id="RP-ENV",
        hypothesis_id="H999",
        plan_id="PLAN-H999",
        candidate_models=["no_such_model"],
        reference_models=["persistence"],
        metrics=["MAE"],
        confirmation_criteria=["criterion"],
        falsification_criteria=["criterion"],
        dataset_id="BOILER-REAL",
        dataset_hash="real",
        input_variables=["feature_01"],
        target_variable="main_steam_mass_flow",
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        baseline_models=["persistence"],
    )

    runner = RealSklearnExperimentRunner()

    with pytest.raises(
        ValueError,
        match="unknown_model",
    ):
        runner.run(contract)


def test_convergence_warning_recorded():
    """
    ConvergenceWarning must be recorded in
    ExperimentResult.model_records, not silently lost.
    """

    contract = ExperimentContract(
        experiment_id="EXP-CONV",
        problem_id="RP-ENV",
        hypothesis_id="H999",
        plan_id="PLAN-H999",
        candidate_models=["ridge"],
        reference_models=["persistence"],
        metrics=["MAE"],
        confirmation_criteria=[
            "all_candidates_worse_than_reference_on:MAE"
        ],
        falsification_criteria=[
            "any_candidate_better_than_reference_on:MAE"
        ],
        dataset_id="BOILER-REAL",
        dataset_hash="real",
        input_variables=["feature_01"],
        target_variable="main_steam_mass_flow",
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        baseline_models=["persistence"],
    )

    payload = {
        "experiment_id": "EXP-CONV",
        "status": "completed",
        "dataset": {"sha256": "a" * 64},
        "reference_model": {
            "locked_test_metrics": {
                "mae_t_h": 1.0,
                "rmse_t_h": 1.0,
                "r2": 0.0,
                "mbe_t_h": 0.0,
            },
        },
        "selected_model_by_validation": "ridge",
        "models": {
            "ridge": {
                "fit_success": True,
                "fit_converged": False,
                "warnings": [
                    "ConvergenceWarning: "
                    "max_iter reached"
                ],
                "failure_reason": None,
                "model_config": {"alpha": 1.0},
                "validation_metrics": {},
                "locked_test_metrics": {
                    "mae_t_h": 2.0,
                    "rmse_t_h": 2.0,
                    "r2": -1.0,
                    "mbe_t_h": 1.0,
                },
                "train_samples": 100,
                "validation_samples": 20,
                "test_samples": 20,
                "random_seed": 42,
                "dataset_sha256": "a" * 64,
                "artifact_paths": [],
            },
        },
        "split": {
            "locked_test_used_for_selection": False,
        },
        "completed_at": (
            "2026-08-20T12:00:00+08:00"
        ),
    }

    class StubBackend:
        __test__ = False

        def run(self, backend_contract):
            return payload

    runner = RealSklearnExperimentRunner(
        backend=StubBackend(),
        output_dir=".",
    )

    result, _trace = runner.run(contract)

    record = result.model_records["ridge"]

    assert record.fit_success is True
    assert record.fit_converged is False
    assert "ConvergenceWarning" in record.warnings[0]
    assert record.model_configuration == {"alpha": 1.0}

    # A model that FAILED must fail closed, not be swapped.
    failed_payload = dict(payload)
    failed_models = {
        "ridge": {
            **payload["models"]["ridge"],
            "fit_success": False,
            "failure_reason": (
                "RuntimeError: exploded"
            ),
        },
    }
    failed_payload["models"] = failed_models

    class FailingBackend:
        __test__ = False

        def run(self, backend_contract):
            return failed_payload

    failing_runner = RealSklearnExperimentRunner(
        backend=FailingBackend(),
        output_dir=".",
    )

    with pytest.raises(
        RuntimeError,
        match="model_fit_failed:ridge",
    ):
        failing_runner.run(contract)


def test_execution_backends_abstraction():
    assert issubclass(LocalCPUBackend, ExecutionBackend)
    assert issubclass(RemoteCPUBackend, ExecutionBackend)
    assert issubclass(CUDABackend, ExecutionBackend)

    local = get_execution_backend("local_cpu")

    assert local.name == "local_cpu"
    assert "backend" in local.capability_summary()

    remote = get_execution_backend("remote_cpu")

    with pytest.raises(
        NotImplementedError,
        match="remote_cpu_backend_placeholder",
    ):
        remote.execute(None)

    with pytest.raises(ValueError):
        get_execution_backend("quantum")


def test_status_matrix_has_adapter_status():
    rows = model_status_matrix(
        capability=ExperimentCapabilityRegistry(
            environment=make_fake_env(torch_available=False)
        )
    )

    by_name = {
        row["model_name"]: row
        for row in rows
    }

    assert by_name["ridge"]["adapter_status"] == (
        "RUNNER_CALLABLE"
    )

    assert by_name["transformer"]["adapter_status"] in {
        "MISSING_DEPENDENCY",
        "SOURCE_ONLY",
    }

    assert by_name["candidate_xgboost_direct_volume_20m"][
        "adapter_status"
    ] == "MISSING_DEPENDENCY"
