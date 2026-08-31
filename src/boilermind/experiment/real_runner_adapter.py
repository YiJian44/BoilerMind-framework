from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boilermind.audit.execution_trace import (
    ExperimentExecutionTrace,
)

from boilermind.core.contracts.experiment import (
    ExperimentContract,
    ExperimentResult,
    ModelExperimentRecord,
)

from boilermind.core.enums import ExperimentStatus
from boilermind.experiment.metric_normalizer import (
    normalize_metrics,
    numeric_normalized_metrics,
)

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.experiment.real_sklearn_backend import (
    RealSklearnExperimentBackend,
)

from boilermind.models import (
    build_default_registry,
)


_METRIC_KEY_MAP = {
    "MAE": "mae_t_h",
    "RMSE": "rmse_t_h",
    "R2": "r2",
    "MBE": "mbe_t_h",
}


class RealSklearnExperimentRunner:
    """
    Adapter implementing the Contracts-chain runner interface:

        run(contract: ExperimentContract)
            -> (ExperimentResult, ExperimentExecutionTrace)

    It reuses the existing real_sklearn backend unchanged.

    Protocol guarantees (fail closed):
    - candidate set is frozen by the contract before execution
    - validation is the ONLY selection scope
    - locked test is evaluation only
    - every predeclared candidate (and persistence) receives
      its own locked-test metrics in ExperimentResult
    """

    is_test_only = False

    DEFAULT_OUTPUT_DIR = (
        Path(__file__).resolve().parents[3]
        / "runtime"
        / "experiment_runs"
    )

    def __init__(
        self,
        *,
        registry: ExperimentCapabilityRegistry | None = None,
        backend: RealSklearnExperimentBackend | None = None,
        model_registry: Any | None = None,
        output_dir: str | Path | None = None,
        torch_backend: Any | None = None,
    ):

        self.registry = (
            registry or ExperimentCapabilityRegistry()
        )

        self.backend = (
            backend or RealSklearnExperimentBackend()
        )

        self.model_registry = (
            model_registry
            or build_default_registry()
        )

        self.torch_backend = torch_backend

        self.output_dir = Path(
            output_dir
            or os.environ.get(
                "BOILERMIND_EXPERIMENT_OUTPUT_DIR",
                self.DEFAULT_OUTPUT_DIR,
            )
        )

    # ---------------------------------------------------------
    # Contract translation
    # ---------------------------------------------------------

    def _backend_contract(
        self,
        contract: ExperimentContract,
    ) -> dict[str, Any]:

        candidate_models = list(
            contract.candidate_models
        )

        if not candidate_models:
            raise ValueError(
                "candidate_models_required"
            )

        # Resolve every candidate through ModelRegistry ->
        # Adapter. No model-name if/elif branches here.
        for model_id in candidate_models:
            try:
                self.model_registry.get(model_id)
                self.model_registry.build_adapter(
                    model_id
                )
            except KeyError as exc:
                raise ValueError(
                    "unknown_model:"
                    f"{model_id}"
                ) from exc
            except Exception as exc:
                raise ValueError(
                    "adapter_unavailable:"
                    f"{model_id}:{exc}"
                ) from exc

        unknown = [
            model_id
            for model_id in candidate_models
            if model_id
            not in self.registry.available_models()
        ]

        if unknown:
            raise ValueError(
                "candidate_model_not_enabled:"
                + ",".join(unknown)
            )

        unsupported_metrics = [
            metric
            for metric in contract.metrics
            if metric not in self.registry.metrics()
        ]

        if unsupported_metrics:
            raise ValueError(
                "unsupported_metrics:"
                + ",".join(unsupported_metrics)
            )

        if not self.registry.dataset_exists():
            raise FileNotFoundError(
                self.registry.dataset_path
            )

        return {
            "dataset_path": str(
                self.registry.dataset_path.resolve()
            ),
            "experiment_id": contract.experiment_id,
            "output_dir": str(self.output_dir),
            "window_steps": (
                contract.window_steps
                or self.registry.window_steps
            ),
            "prediction_horizon_steps": (
                contract.prediction_horizon_steps
                or self.registry.prediction_horizon_steps
            ),
            "sampling_interval_seconds": (
                contract.sampling_interval_seconds
                or self.registry.sampling_interval_seconds
            ),
            "train_ratio": self.registry.train_ratio,
            "validation_ratio": (
                self.registry.validation_ratio
            ),
            "model_candidates": candidate_models,
            "random_seed": int(contract.random_seed),
            "hypothesis_id": contract.hypothesis_id,
            "plan_id": contract.plan_id,
        }

    # ---------------------------------------------------------
    # Payload -> contract mapping
    # ---------------------------------------------------------

    def _normalize_locked_metrics(
        self,
        payload_metrics: dict[str, Any],
    ) -> dict[str, float]:

        normalized: dict[str, float] = {}

        for metric_name, payload_key in (
            _METRIC_KEY_MAP.items()
        ):
            if payload_key not in payload_metrics:
                raise ValueError(
                    "locked_test_metric_missing:"
                    f"{payload_key}"
                )

            normalized[metric_name] = float(
                payload_metrics[payload_key]
            )

        return normalized

    def _to_result(
        self,
        contract: ExperimentContract,
        payload: dict[str, Any],
    ) -> ExperimentResult:

        if payload.get("experiment_id") != (
            contract.experiment_id
        ):
            raise ValueError(
                "experiment_id_mismatch_between_"
                "contract_and_backend"
            )

        models = payload.get("models", {})

        if not isinstance(models, dict) or not models:
            raise ValueError(
                "backend_payload_missing_models"
            )

        # Hard guarantee: the runner must execute EXACTLY the
        # models frozen in the contract. No reselection, no
        # default pool, no substitution.
        executed_models = set(models.keys())
        expected_models = set(contract.candidate_models)

        if executed_models != expected_models:
            raise ValueError(
                "executed_models_mismatch_contract:"
                f"executed={sorted(executed_models)};"
                f"contract={sorted(expected_models)}"
            )

        # Standardized per-model records.
        model_records: dict[
            str,
            ModelExperimentRecord,
        ] = {}

        for model_id, model_result in models.items():
            locked = model_result.get(
                "locked_test_metrics",
                {},
            )

            model_records[model_id] = (
                ModelExperimentRecord(
                    model_name=model_id,
                    fit_success=bool(
                        model_result.get(
                            "fit_success",
                            False,
                        )
                    ),
                    fit_converged=bool(
                        model_result.get(
                            "fit_converged",
                            False,
                        )
                    ),
                    warnings=list(
                        model_result.get(
                            "warnings",
                            [],
                        )
                    ),
                    failure_reason=(
                        model_result.get(
                            "failure_reason"
                        )
                    ),
                    runtime_seconds=(
                        model_result.get(
                            "elapsed_seconds"
                        )
                    ),
                    model_configuration=dict(
                        model_result.get(
                            "model_config",
                            {},
                        )
                        or model_result.get(
                            "selected_parameters",
                            {},
                        )
                    ),
                    validation_metrics=dict(
                        model_result.get(
                            "validation_metrics",
                            {},
                        )
                    ),
                    locked_test_metrics=(
                        self._normalize_locked_metrics(
                            locked
                        )
                        if locked
                        else {}
                    ),
                    train_samples=(
                        model_result.get(
                            "train_samples"
                        )
                    ),
                    validation_samples=(
                        model_result.get(
                            "validation_samples"
                        )
                    ),
                    test_samples=(
                        model_result.get(
                            "test_samples"
                        )
                    ),
                    random_seed=(
                        model_result.get(
                            "random_seed"
                        )
                    ),
                    dataset_sha256=(
                        model_result.get(
                            "dataset_sha256"
                        )
                        or payload.get(
                            "dataset",
                            {},
                        ).get("sha256")
                    ),
                    artifact_paths=list(
                        model_result.get(
                            "artifact_paths",
                            [],
                        )
                    ),
                    device=model_result.get("device"),
                    epochs_completed=model_result.get("epochs_completed"),
                    best_epoch=model_result.get("best_epoch"),
                    training_loss=model_result.get("training_loss"),
                    validation_loss=model_result.get("validation_loss"),
                )
            )

        failed_models = [
            model_id
            for model_id, record in (
                model_records.items()
            )
            if not record.fit_success
        ]

        if failed_models:
            failed = failed_models[0]
            raise RuntimeError(
                "model_fit_failed:"
                f"{failed}:"
                f"{model_records[failed].failure_reason}"
            )

        candidate_locked_test_metrics = {}

        for model_id, record in (
            model_records.items()
        ):
            if record.locked_test_metrics:
                candidate_locked_test_metrics[model_id] = (
                    record.locked_test_metrics
                )

        reference = payload.get(
            "reference_model",
            {},
        )

        reference_locked = reference.get(
            "locked_test_metrics",
            {},
        )

        candidate_locked_test_metrics[
            self.registry.reference_model
        ] = self._normalize_locked_metrics(
            reference_locked
        )

        selected_model = payload.get(
            "selected_model_by_validation"
        )

        if (
            selected_model is None
            or selected_model not in candidate_locked_test_metrics
        ):
            raise ValueError(
                "selected_model_result_missing"
            )

        selected_metrics = (
            candidate_locked_test_metrics[
                selected_model
            ]
        )

        baseline_metrics = (
            candidate_locked_test_metrics[
                self.registry.reference_model
            ]
        )

        split = payload.get("split", {})

        locked_test_used_for_selection = bool(
            split.get(
                "locked_test_used_for_selection",
                True,
            )
        )

        if locked_test_used_for_selection:
            raise ValueError(
                "locked_test_used_for_selection"
            )

        artifacts = []

        result_artifact = payload.get(
            "result_artifact"
        )

        if result_artifact:
            artifacts.append(
                str(result_artifact)
            )

        for model_result in models.values():
            for key in (
                "model_artifact",
                "prediction_artifact",
            ):
                artifact = model_result.get(key)

                if artifact:
                    artifacts.append(
                        str(artifact)
                    )

        completed_at = payload.get(
            "completed_at"
        )

        started_at = datetime.now(timezone.utc)

        completed_dt = (
            datetime.fromisoformat(completed_at)
            if isinstance(completed_at, str)
            else datetime.now(timezone.utc)
        )

        execution_notes = [
            "REAL_SKLEARN_EXECUTION",
            "chronological_train_validation_locked_test",
            "validation_only_model_selection",
            "locked_test_not_used_for_selection",
            (
                "selected_model_by_validation:"
                f"{selected_model}"
            ),
            (
                "candidate_locked_test_models:"
                + ",".join(
                    sorted(
                        candidate_locked_test_metrics
                    )
                )
            ),
            (
                "dataset_sha256:"
                f"{payload.get('dataset', {}).get('sha256')}"
            ),
        ]

        return ExperimentResult(
            experiment_id=contract.experiment_id,
            problem_id=contract.problem_id,
            hypothesis_id=contract.hypothesis_id,
            plan_id=contract.plan_id,
            status=ExperimentStatus.COMPLETED,
            metrics={**selected_metrics, **numeric_normalized_metrics(selected_metrics)},
            raw_metrics=selected_metrics,
            normalized_metrics=normalize_metrics(selected_metrics),
            baseline_metrics=baseline_metrics,
            candidate_locked_test_metrics=(
                candidate_locked_test_metrics
            ),
            model_records=model_records,
            artifacts=list(
                dict.fromkeys(artifacts)
            ),
            execution_notes=execution_notes,
            started_at=started_at,
            completed_at=completed_dt,
        )

    def _to_trace(
        self,
        contract: ExperimentContract,
        payload: dict[str, Any],
    ) -> ExperimentExecutionTrace:

        return ExperimentExecutionTrace(
            experiment_id=contract.experiment_id,
            dataset_frozen=True,
            leakage_check_passed=True,
            baseline_valid=True,
            metric_check_passed=True,
            notes=[
                "REAL_SKLEARN_EXECUTION_TRACE",
                (
                    "dataset_sha256:"
                    f"{payload.get('dataset', {}).get('sha256')}"
                ),
                "locked_test_used_for_selection:False",
            ],
        )

    # ---------------------------------------------------------
    # Runner interface
    # ---------------------------------------------------------

    def run(
        self,
        contract: ExperimentContract,
    ) -> tuple[
        ExperimentResult,
        ExperimentExecutionTrace,
    ]:

        if not isinstance(contract, ExperimentContract):
            raise TypeError(
                "experiment_contract_required"
            )

        backend_contract = self._backend_contract(
            contract
        )

        frameworks = {
            self.model_registry.get(name).framework
            for name in contract.candidate_models
        }
        if frameworks == {"torch"}:
            if self.torch_backend is None:
                from boilermind.experiment.torch_backend import TorchExperimentBackend
                self.torch_backend = TorchExperimentBackend(model_registry=self.model_registry)
            payload = self.torch_backend.run(backend_contract)
        elif "torch" in frameworks:
            raise ValueError("mixed_sklearn_torch_contract_not_supported")
        else:
            payload = self.backend.run(backend_contract)

        if not isinstance(payload, dict):
            raise TypeError(
                "backend_must_return_dict"
            )

        result = self._to_result(
            contract,
            payload,
        )

        trace = self._to_trace(
            contract,
            payload,
        )

        return result, trace

    def execute(
        self,
        contract: ExperimentContract,
    ) -> tuple[
        ExperimentResult,
        ExperimentExecutionTrace,
    ]:
        """
        Business-layer entry point (unified runner interface).
        """

        return self.run(contract)
