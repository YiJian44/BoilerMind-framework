from __future__ import annotations

import numpy as np
import pytest

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.experiment.real_runner_adapter import RealSklearnExperimentRunner
from boilermind.experiment.time_series_data import DatasetBuilder, TimeSeriesDataContract
from boilermind.models.adapters import TorchModelAdapter
from boilermind.models.catalog import build_default_registry
from boilermind.models.execution_environment import ExecutionEnvironment
from boilermind.models.torch_factory import get_torch_architecture_factory


def env(torch=True, cuda=False):
    return ExecutionEnvironment(os="test", python_version="3.11", sklearn_available=True,
        torch_available=torch, cuda_available=cuda, gpu_available=cuda)


def contract():
    return TimeSeriesDataContract(tuple(range(2)), (2,), 15, 4, 2, .6, .2)


def test_torch_adapter_interface_and_factory_resolution():
    registry = build_default_registry()
    for name in ("lstm", "gru", "transformer", "dlinear"):
        spec = registry.get(name)
        adapter = registry.build_adapter(name)
        assert isinstance(adapter, TorchModelAdapter)
        assert callable(adapter.fit) and callable(adapter.predict) and callable(adapter.evaluate)
        assert spec.architecture_factory == name
        assert callable(get_torch_architecture_factory(name))


def test_torch_unavailable_and_device_rules_fail_closed(monkeypatch):
    adapter = build_default_registry().build_adapter("lstm")
    monkeypatch.setattr(
        adapter,
        "_load_torch",
        lambda: (_ for _ in ()).throw(
            RuntimeError("torch_required:torch_not_installed:lstm")
        ),
    )
    with pytest.raises(RuntimeError, match="torch_required:torch_not_installed"):
        adapter.fit(np.zeros((2, 4, 2)), np.zeros(2))

    class CUDA:
        @staticmethod
        def is_available(): return False
    class FakeTorch:
        cuda = CUDA()
        @staticmethod
        def device(value): return value
    assert adapter._select_device(FakeTorch) == "cpu"
    cuda_spec = adapter.spec.model_copy(update={"requires_cuda": True, "cpu_supported": False})
    with pytest.raises(RuntimeError, match="cuda_required_but_unavailable"):
        TorchModelAdapter(cuda_spec)._select_device(FakeTorch)


def test_source_training_ignores_old_checkpoint_but_reuse_rejects_it():
    spec = build_default_registry().get("lstm").model_copy(update={
        "checkpoint_available": True,
        "checkpoint_compatibility": {"compatible": False, "mismatches": ["features"]},
        "checkpoint_inference_supported": True,
    })
    assert isinstance(TorchModelAdapter(spec, reuse_checkpoint=False), TorchModelAdapter)
    with pytest.raises(ValueError, match="checkpoint_incompatible_reuse_refused"):
        TorchModelAdapter(spec, reuse_checkpoint=True)


def test_chronological_split_scaler_train_only_and_locked_isolation():
    values = np.arange(90, dtype=float).reshape(30, 3)
    values[20:, :2] += 10000
    built = DatasetBuilder().build(values, contract())
    assert built.source_indices["train"].max() < built.source_indices["validation"].min()
    assert built.source_indices["validation"].max() < built.source_indices["locked_test"].min()
    assert built.X_validation.max() > 1.0

    changed = values.copy()
    first_locked_row = int(built.source_indices["locked_test"].min())
    changed[first_locked_row:, :] += 999999
    rebuilt = DatasetBuilder().build(changed, contract())
    np.testing.assert_allclose(built.X_train, rebuilt.X_train)
    np.testing.assert_allclose(built.X_validation[:-1], rebuilt.X_validation[:-1])


def test_registry_capability_gpr_and_unknown_deep_model():
    capability = ExperimentCapabilityRegistry(environment=env(torch=True))
    assert {"lstm", "gru", "transformer", "dlinear", "gpr"}.issubset(capability.available_models())
    with pytest.raises(KeyError, match="unknown_model"):
        build_default_registry().build_adapter("unknown_deep")


def test_contract_runner_dispatches_to_torch_backend_and_standard_record():
    registry = build_default_registry()
    capability = ExperimentCapabilityRegistry(environment=env(torch=True))

    class StubTorchBackend:
        def run(self, payload):
            metrics = {"mae_t_h": 1.0, "rmse_t_h": 1.0, "r2": 0.0, "mbe_t_h": 0.0}
            return {"experiment_id": payload["experiment_id"], "dataset": {"sha256": "a" * 64},
                "models": {"lstm": {"fit_success": True, "fit_converged": True,
                    "validation_metrics": metrics, "locked_test_metrics": metrics,
                    "elapsed_seconds": .1, "random_seed": 42, "device": "cpu",
                    "epochs_completed": 2, "best_epoch": 1, "training_loss": .2, "validation_loss": .3}},
                "selected_model_by_validation": "lstm", "reference_model": {"locked_test_metrics": metrics},
                "split": {"locked_test_used_for_selection": False}}

    runner = RealSklearnExperimentRunner(registry=capability, model_registry=registry,
        torch_backend=StubTorchBackend(), output_dir=".")
    c = ExperimentContract(experiment_id="EXP-TORCH", problem_id="P", hypothesis_id="H", plan_id="PL",
        dataset_id="D", dataset_hash="hash", input_variables=["feature_01"], target_variable="main_steam_mass_flow",
        train_split="train", validation_split="validation", test_split="locked", baseline_models=["persistence"],
        candidate_models=["lstm"], metrics=["MAE"], confirmation_criteria=["c"], falsification_criteria=["f"])
    result, trace = runner.run(c)
    record = result.model_records["lstm"]
    assert record.device == "cpu" and record.epochs_completed == 2
    assert trace.leakage_check_passed is True
