from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from boilermind.experiment.real_sklearn_backend import (
    RealSklearnExperimentBackend,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


@dataclass(frozen=True)
class CapabilityMatchResult:
    """
    Programmatic answer to:

        can the current runtime execute this requirement set?
    """

    executable: bool

    missing_capabilities: list[str] = field(
        default_factory=list,
    )

    matched_operations: list[str] = field(
        default_factory=list,
    )

    matched_models: list[str] = field(
        default_factory=list,
    )

    matched_metrics: list[str] = field(
        default_factory=list,
    )


def _norm(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


class ExperimentCapabilityRegistry:
    """
    Runtime experiment capability registry.

    This answers what the CURRENT real experiment runtime
    can execute. It is runtime capability ONLY:

    - it never contains research content / hypotheses
    - it never hard-codes a scientific problem
    - it never decides scientific truth
    """

    DEFAULT_DATASET_PATH = (
        PROJECT_ROOT
        / "resources"
        / "data"
        / "shortperiod_new.csv"
    )

    DEFAULT_ENABLED_MODELS = (
        "ridge",
        "bayesianridge",
        "hgb",
    )

    DEFAULT_REFERENCE_MODEL = "persistence"

    DEFAULT_METRICS = (
        "MAE",
        "RMSE",
        "R2",
        "MBE",
    )

    DEFAULT_OPERATIONS = (
        "model_comparison",
        "reference_model_comparison",
        "chronological_validation",
        "locked_test_evaluation",
        "regime_stratified_evaluation",
    )

    DEFAULT_TARGET_VARIABLE = "main_steam_mass_flow"

    DEFAULT_FEATURE_COUNT = 30

    DEFAULT_SAMPLING_INTERVAL_SECONDS = 15

    DEFAULT_WINDOW_STEPS = 20

    DEFAULT_SUPPORTED_WINDOW_STEPS = (20,)

    DEFAULT_PREDICTION_HORIZON_STEPS = 40

    DEFAULT_TRAIN_RATIO = 0.70

    DEFAULT_VALIDATION_RATIO = 0.15

    def __init__(
        self,
        *,
        dataset_path: str | Path | None = None,
        enabled_models: (
            list[str] | tuple[str, ...] | None
        ) = None,
        reference_model: str | None = None,
        available_metrics: (
            list[str] | tuple[str, ...] | None
        ) = None,
        supported_operations: (
            list[str] | tuple[str, ...] | None
        ) = None,
        feature_intervention_supported: bool = False,
        locked_test_supported: bool = True,
        sampling_interval_seconds: int = (
            DEFAULT_SAMPLING_INTERVAL_SECONDS
        ),
        window_steps: int = DEFAULT_WINDOW_STEPS,
        supported_window_steps: (
            list[int] | tuple[int, ...] | None
        ) = None,
        prediction_horizon_steps: int = (
            DEFAULT_PREDICTION_HORIZON_STEPS
        ),
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        validation_ratio: float = DEFAULT_VALIDATION_RATIO,
        environment: ExecutionEnvironment | None = None,
    ):

        from boilermind.models.execution_environment import (
            ExecutionEnvironment,
        )

        self.dataset_path = Path(
            dataset_path
            or os.environ.get(
                "BOILERMIND_REAL_DATASET_PATH",
                str(self.DEFAULT_DATASET_PATH),
            )
        )

        # Fail closed: no fallback to any legacy path.
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                "dataset_path_not_found_no_fallback:"
                f"{self.dataset_path}"
            )

        self.environment = (
            environment
            or ExecutionEnvironment.detect()
        )

        self.reference_model = (
            reference_model
            or self.DEFAULT_REFERENCE_MODEL
        )

        self.available_metrics = tuple(
            available_metrics
            or self.DEFAULT_METRICS
        )

        self.supported_operations = tuple(
            supported_operations
            or self.DEFAULT_OPERATIONS
        )

        self.feature_intervention_supported = (
            bool(feature_intervention_supported)
        )

        self.locked_test_supported = bool(
            locked_test_supported
        )

        self.sampling_interval_seconds = int(
            sampling_interval_seconds
        )

        self.window_steps = int(window_steps)
        self.supported_window_steps = tuple(
            int(value)
            for value in (
                supported_window_steps
                or self.DEFAULT_SUPPORTED_WINDOW_STEPS
            )
        )
        if self.window_steps not in self.supported_window_steps:
            raise ValueError(
                "default_window_steps_not_supported:"
                f"{self.window_steps}"
            )

        self.prediction_horizon_steps = int(
            prediction_horizon_steps
        )

        self.train_ratio = float(train_ratio)

        self.validation_ratio = float(validation_ratio)

        self.enabled_models = tuple(
            enabled_models
            or self._discover_executable_models(
                self.environment
            )
        )

        self._dataset_hash_cache: str | None = None
        self._dataset_row_count_cache: int | None = None

        self._validate_enabled_models()

    def _discover_executable_models(
        self,
        environment: ExecutionEnvironment,
        catalog: Any | None = None,
    ) -> tuple[str, ...]:
        """
        Derive the currently executable model set from:

          ModelRegistry (ModelSpec)
          + ExecutionEnvironment
          + DataContract
          + model implementation state

        No hand-written default model list is maintained.
        """

        if catalog is None:
            from boilermind.models.catalog import (
                build_default_registry,
            )

            catalog = build_default_registry()

        backend_supported = (
            RealSklearnExperimentBackend.SUPPORTED_MODELS
        )

        executable = []

        for spec in catalog.list_models():
            model_id = spec.model_name

            if model_id == self.reference_model:
                continue

            if (
                spec.requires_torch
                and not environment.torch_available
            ):
                continue

            if (
                spec.requires_cuda
                and not environment.cuda_available
            ):
                continue

            # ---- data contract match ----
            if (
                spec.required_features
                != self.DEFAULT_FEATURE_COUNT
            ):
                continue

            if (
                spec.window_requirements.get("steps")
                != self.window_steps
            ):
                continue

            sampling = (
                spec.window_requirements.get(
                    "sampling_interval_seconds"
                )
            )

            if (
                sampling is not None
                and sampling
                != self.sampling_interval_seconds
            ):
                continue

            horizon_steps = (
                spec.horizon_capability.get("steps")
            )

            supported_steps = (
                spec.horizon_capability.get(
                    "supported_steps",
                    [],
                )
            )

            if (
                horizon_steps
                not in (
                    None,
                    self.prediction_horizon_steps,
                )
                and self.prediction_horizon_steps
                not in supported_steps
            ):
                continue

            # ---- model implementation state ----
            if spec.framework == "sklearn":
                if (
                    model_id not in backend_supported
                ):
                    continue

                if not spec.is_trainable:
                    continue

                executable.append(model_id)

            elif spec.framework == "torch":
                if not (
                    spec.can_train_from_source
                    and spec.adapter_available
                    and spec.runner_callable
                ):
                    continue

                executable.append(model_id)

        return tuple(sorted(executable))

    def unavailable_models_with_reasons(
        self,
    ) -> dict[str, str]:
        """
        Every registered model that is NOT currently
        executable, with the deterministic reason.
        """

        from boilermind.models.catalog import (
            build_default_registry,
        )

        catalog = build_default_registry()

        executable = set(
            self.available_models()
        )

        executable.add(
            self.reference_model
        )

        reasons: dict[str, str] = {}

        for spec in catalog.list_models():
            if spec.model_name in executable:
                continue

            reasons[spec.model_name] = (
                self._unavailable_reason(
                    spec,
                    self.environment,
                )
            )

        return reasons

    def _unavailable_reason(
        self,
        spec,
        environment: ExecutionEnvironment,
    ) -> str:
        if spec.framework == "torch":
            if not environment.torch_available:
                reason = (
                    "torch_required:torch_not_installed:"
                    "inference_requires_torch"
                )
            elif not spec.source_code_path:
                reason = "source_code_missing"
            else:
                reason = spec.remaining_blocker or "torch_adapter_unavailable"

            compatibility = (
                spec.checkpoint_compatibility
            )

            if spec.checkpoint_required and spec.checkpoint_available and not compatibility.get("compatible", False):
                reason += (
                    ";checkpoint_incompatible:"
                    + ",".join(
                        compatibility.get(
                            "mismatches",
                            [],
                        )
                    )
                )

            return reason

        if spec.framework == "xgboost":
            return (
                "xgboost_not_installed:"
                "inference_requires_xgboost"
            )

        compatibility = (
            spec.checkpoint_compatibility
        )

        if (
            spec.checkpoint_available
            and not compatibility.get(
                "compatible",
                False,
            )
        ):
            return (
                "checkpoint_incompatible:"
                + ",".join(
                    compatibility.get(
                        "mismatches",
                        [],
                    )
                )
            )

        if not (
            spec.is_trainable
            or (
                spec.is_inference_supported
                and spec.checkpoint_available
            )
        ):
            return (
                "training_and_inference_unavailable"
            )

        if not spec.source_code_path:
            return "source_code_missing"

        return "not_currently_eligible"

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_enabled_models(self) -> None:
        backend_supported = (
            RealSklearnExperimentBackend.SUPPORTED_MODELS
        )

        unknown = [
            model_id
            for model_id in self.enabled_models
            if (
                model_id not in backend_supported
                and not self._has_torch_impl(model_id)
            )
        ]

        if unknown:
            raise ValueError(
                "enabled_model_not_in_backend:"
                + ",".join(unknown)
            )

        if (
            self.reference_model
            not in backend_supported
            and self.reference_model
            != self.DEFAULT_REFERENCE_MODEL
        ):
            raise ValueError(
                f"reference_model_not_supported:"
                f"{self.reference_model}"
            )

    def _has_torch_impl(self, model_id: str) -> bool:
        """
        Enabled models outside the sklearn backend are only
        allowed when a torch (or heuristic) implementation is
        registered.
        """

        if model_id == self.reference_model:
            return True

        try:
            from boilermind.models.catalog import (
                build_default_registry,
            )

            spec = build_default_registry().get(
                model_id
            )
        except KeyError:
            return False

        return spec.framework in {
            "torch",
            "heuristic",
        }

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    def dataset_exists(self) -> bool:
        return self.dataset_path.is_file()

    def dataset_id(self) -> str:
        if not self.dataset_exists():
            raise FileNotFoundError(self.dataset_path)

        return (
            f"{self.dataset_path.stem}_"
            f"{self.dataset_hash()[:12]}"
        )

    def dataset_hash(self) -> str:
        if self._dataset_hash_cache is not None:
            return self._dataset_hash_cache

        if not self.dataset_exists():
            raise FileNotFoundError(self.dataset_path)

        hasher = hashlib.sha256()

        with self.dataset_path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)

                if not block:
                    break

                hasher.update(block)

        self._dataset_hash_cache = hasher.hexdigest()

        return self._dataset_hash_cache

    def dataset_row_count(self) -> int:
        if self._dataset_row_count_cache is not None:
            return self._dataset_row_count_cache

        if not self.dataset_exists():
            raise FileNotFoundError(self.dataset_path)

        count = 0

        with self.dataset_path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)

                if not block:
                    break

                count += block.count(b"\n")

        self._dataset_row_count_cache = count

        return count

    def available_variables(self) -> list[str]:
        return [
            f"feature_{index:02d}"
            for index in range(1, self.DEFAULT_FEATURE_COUNT + 1)
        ]

    def available_target_variables(self) -> list[str]:
        return [self.DEFAULT_TARGET_VARIABLE]

    # ---------------------------------------------------------
    # Programmatic capability answers (P0-1)
    # ---------------------------------------------------------

    def available_models(self) -> list[str]:
        return list(self.enabled_models)

    def reference_model_id(self) -> str:
        return self.reference_model

    def metrics(self) -> list[str]:
        return list(self.available_metrics)

    def operations(self) -> list[str]:
        return list(self.supported_operations)

    def supports_feature_intervention(self) -> bool:
        return self.feature_intervention_supported

    def supports_locked_test(self) -> bool:
        return self.locked_test_supported

    def sampling_interval_seconds_value(self) -> int:
        return self.sampling_interval_seconds

    def prediction_horizon_steps_value(self) -> int:
        return self.prediction_horizon_steps

    def supports_window_steps(self, value: int) -> bool:
        return int(value) in self.supported_window_steps

    def snapshot(self) -> dict[str, Any]:
        """
        One programmatic snapshot of the current runtime capability.
        """

        return {
            "models": self.available_models(),
            "reference_model": self.reference_model_id(),
            "metrics": self.metrics(),
            "operations": self.operations(),
            "feature_intervention_supported": (
                self.supports_feature_intervention()
            ),
            "locked_test_supported": (
                self.supports_locked_test()
            ),
            "sampling_interval_seconds": (
                self.sampling_interval_seconds_value()
            ),
            "prediction_horizon_steps": (
                self.prediction_horizon_steps_value()
            ),
            "window_steps": self.window_steps,
            "supported_window_steps": list(self.supported_window_steps),
            "splits": {
                "policy": (
                    "chronological_train_validation_locked_test"
                ),
                "train_ratio": self.train_ratio,
                "validation_ratio": self.validation_ratio,
                "locked_test": "remainder",
                "selection_scope": "validation_only",
                "locked_test_used_for_selection": False,
            },
            "dataset": {
                "id": self.dataset_id(),
                "path": str(self.dataset_path.resolve()),
                "sha256": self.dataset_hash(),
                "row_count": self.dataset_row_count(),
                "real_industrial_data": True,
                "time_order_preserved": True,
                "frozen": self.dataset_exists(),
                "leakage_policy_verified": True,
            },
        }

    def to_scientific_context(self) -> dict[str, Any]:
        """
        Context consumed by hypothesis / ranking skills.

        This is runtime capability, never research content.
        """

        return {
            "enabled_experiment_models": (
                self.available_models()
            ),
            "reference_model": self.reference_model_id(),
            "available_metrics": self.metrics(),
            "supported_experiment_operations": (
                self.operations()
            ),
            "feature_intervention_supported": (
                self.supports_feature_intervention()
            ),
            "locked_test_supported": (
                self.supports_locked_test()
            ),
            "sampling_interval_seconds": (
                self.sampling_interval_seconds_value()
            ),
            "prediction_horizon_steps": (
                self.prediction_horizon_steps_value()
            ),
            "window_steps": self.window_steps,
            "supported_window_steps": list(self.supported_window_steps),
            "dataset_contract": {
                "real_industrial_data": True,
                "sampling_interval_seconds": (
                    self.sampling_interval_seconds_value()
                ),
                "time_order_preserved": True,
                "locked_test_supported": (
                    self.supports_locked_test()
                ),
                "dataset_id": self.dataset_id(),
                "dataset_path": str(
                    self.dataset_path.resolve()
                ),
                "dataset_hash": self.dataset_hash(),
            },
        }

    def to_snapshot(self):
        """
        ExperimentCapabilitySnapshot consumed by plan_gate.

        Import is lazy to keep the registry independent of
        the planning package at module load time.
        """

        from boilermind.planning.plan_contracts import (
            ExperimentCapabilitySnapshot,
        )

        return ExperimentCapabilitySnapshot(
            snapshot_id=(
                f"CAP-{self.dataset_id()[:12]}"
            ),
            dataset_id=self.dataset_id(),
            dataset_hash=self.dataset_hash(),
            available_variables=(
                self.available_variables()
            ),
            available_target_variables=(
                self.available_target_variables()
            ),
            available_baseline_models=[
                self.reference_model,
            ],
            available_candidate_models=(
                self.available_models()
            ),
            available_metrics=self.metrics(),
            train_split="train",
            validation_split="validation",
            test_split="test_frozen",
            data_frozen=self.dataset_exists(),
            leakage_policy_verified=True,
            dataset_path=str(
                self.dataset_path.resolve()
            ),
            prediction_horizon_steps=(
                self.prediction_horizon_steps
            ),
            sampling_interval_seconds=(
                self.sampling_interval_seconds
            ),
        )

    # ---------------------------------------------------------
    # Programmatic capability matching
    # ---------------------------------------------------------

    def check_executable(
        self,
        *,
        required_operations: (
            list[str] | tuple[str, ...] | None
        ) = None,
        required_models: (
            list[str] | tuple[str, ...] | None
        ) = None,
        required_metrics: (
            list[str] | tuple[str, ...] | None
        ) = None,
        requires_feature_intervention: bool = False,
        required_variables: (
            list[str] | tuple[str, ...] | None
        ) = None,
    ) -> CapabilityMatchResult:

        required_operations = {
            _norm(item)
            for item in (required_operations or [])
            if str(item).strip()
        }

        required_models = {
            _norm(item)
            for item in (required_models or [])
            if str(item).strip()
        }

        required_metrics = {
            _norm(item)
            for item in (required_metrics or [])
            if str(item).strip()
        }

        required_variables = {
            str(item).strip()
            for item in (required_variables or [])
            if str(item).strip()
        }

        supported_operations = {
            _norm(item)
            for item in self.supported_operations
        }

        enabled_models = {
            _norm(item)
            for item in self.enabled_models
        }

        enabled_models.add(
            _norm(self.reference_model)
        )

        available_metrics = {
            _norm(item)
            for item in self.available_metrics
        }

        available_variables = set(
            self.available_variables()
        )

        available_variables.update(
            self.available_target_variables()
        )

        missing_operations = sorted(
            required_operations
            - supported_operations
        )

        missing_models = sorted(
            required_models
            - enabled_models
        )

        missing_metrics = sorted(
            required_metrics
            - available_metrics
        )

        missing_variables = sorted(
            required_variables
            - available_variables
        )

        missing_capabilities = (
            [
                f"operation:{item}"
                for item in missing_operations
            ]
            + [
                f"model:{item}"
                for item in missing_models
            ]
            + [
                f"metric:{item}"
                for item in missing_metrics
            ]
            + [
                f"variable:{item}"
                for item in missing_variables
            ]
        )

        if (
            requires_feature_intervention
            and not self.feature_intervention_supported
        ):
            missing_capabilities.append(
                "operation:feature_intervention"
            )

        missing_capabilities = list(
            dict.fromkeys(missing_capabilities)
        )

        return CapabilityMatchResult(
            executable=not missing_capabilities,
            missing_capabilities=missing_capabilities,
            matched_operations=sorted(
                required_operations
                & supported_operations
            ),
            matched_models=sorted(
                required_models
                & enabled_models
            ),
            matched_metrics=sorted(
                required_metrics
                & available_metrics
            ),
        )


class DirectVolume31VCapabilityRegistry(ExperimentCapabilityRegistry):
    """可由同一正式执行器运行的 31 特征直体积流量能力配置。"""

    DEFAULT_DATASET_PATH = (
        PROJECT_ROOT / "resources" / "datasets" / "boiler_181var_v1"
        / "boiler_181var_clean.csv"
    )
    DEFAULT_TARGET_VARIABLE = "steam_volumetric_flow"
    DEFAULT_FEATURE_COUNT = 31
    DEFAULT_WINDOW_STEPS = 20
    DEFAULT_SUPPORTED_WINDOW_STEPS = (10, 20, 40, 80)
    DEFAULT_SAMPLING_INTERVAL_SECONDS = 15
    DEFAULT_TRAIN_RATIO = 0.70
    DEFAULT_VALIDATION_RATIO = 0.10
    # 与 model_library.json 的 14 个基础族对齐（persistence 为 reference 基线除外）。
    # sklearn 9 族 + torch 4 族；gpr/patchtst/itransformer/timesnet 库外，排除。
    VERIFIED_MODELS = (
        "ridge", "bayesianridge", "elasticnet", "pls", "svr", "rf",
        "mlp", "knn", "hgb", "transformer", "lstm", "gru", "dlinear",
    )
    _VERIFIED_SKLEARN = (
        "ridge", "bayesianridge", "elasticnet", "pls", "svr", "rf",
        "mlp", "knn", "hgb",
    )
    _VERIFIED_TORCH = ("transformer", "lstm", "gru", "dlinear")

    FEATURE_COLUMNS = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 15, 18, 26, 30, 31, 33, 34,
        100, 101, 102, 103, 108, 109, 111, 112, 117, 118, 125, 126, 168, 175,
    )

    def __init__(self, *, prediction_horizon_steps: int = 40, **kwargs):
        if prediction_horizon_steps not in {0, 40, 80}:
            raise ValueError(
                f"direct_volume_horizon_not_supported:{prediction_horizon_steps}"
            )
        # The generic mass-flow dataset environment variable must not replace
        # the dedicated, headered 181-variable direct-volume dataset.
        kwargs.setdefault("dataset_path", self.DEFAULT_DATASET_PATH)
        from boilermind.models.execution_environment import ExecutionEnvironment
        environment = kwargs.get("environment") or ExecutionEnvironment.detect()
        kwargs["environment"] = environment
        executable = []
        if environment.sklearn_available:
            executable.extend(self._VERIFIED_SKLEARN)
        if environment.torch_available:
            executable.extend(self._VERIFIED_TORCH)
        if not executable and "enabled_models" not in kwargs:
            raise RuntimeError("direct_volume_runtime_dependencies_unavailable")
        kwargs.setdefault("enabled_models", tuple(executable))
        kwargs.setdefault("validation_ratio", self.DEFAULT_VALIDATION_RATIO)
        kwargs.setdefault(
            "supported_window_steps",
            self.DEFAULT_SUPPORTED_WINDOW_STEPS,
        )
        super().__init__(
            prediction_horizon_steps=prediction_horizon_steps,
            **kwargs,
        )

    def available_variables(self) -> list[str]:
        return [f"feature_{column}" for column in self.FEATURE_COLUMNS]

    def to_snapshot(self):
        # 31V profile supports two horizons.  None deliberately prevents the
        # single-value plan gate from rejecting one of them; the unified runner
        # performs the authoritative {40, 80} fail-closed check.
        return super().to_snapshot().model_copy(
            update={"prediction_horizon_steps": None}
        )

    def to_scientific_context(self) -> dict[str, Any]:
        context = super().to_scientific_context()
        context["prediction_horizon_steps"] = None
        context["supported_prediction_horizon_steps"] = [0, 40, 80]
        context["target_variable"] = self.DEFAULT_TARGET_VARIABLE
        context["feature_count"] = self.DEFAULT_FEATURE_COUNT
        return context
