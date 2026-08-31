from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, Field

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)


class ModelSpec(BaseModel):
    """
    Registry metadata for one soft-sensing model.

    The registry distinguishes:
    - scientifically suitable  -> filter_compatible(...)
    - currently executable     -> ExperimentCapabilityRegistry
    """

    model_name: str = Field(min_length=1)

    framework: str = Field(
        min_length=1,
        description=(
            "sklearn | torch | xgboost | heuristic"
        ),
    )

    supported_tasks: list[str] = Field(
        min_length=1,
    )

    task_types: list[str] | None = None

    required_input_type: str = Field(
        min_length=1,
        description=(
            "flattened_window | sequence_window | "
            "raw_window"
        ),
    )

    required_features: int = Field(ge=1)

    feature_indices: list[int] | None = None

    sequence_required: bool

    window_requirements: dict[str, Any] = Field(
        description=(
            "steps / lookback_minutes / "
            "sampling_interval_seconds"
        ),
    )

    horizon_capability: dict[str, Any] = Field(
        description=(
            "steps / minutes / supported_steps"
        ),
    )

    supported_targets: list[str] = Field(
        min_length=1,
    )

    training_available: bool

    trainable: bool | None = None

    inference_available: bool

    inference_supported: bool | None = None

    checkpoint_required: bool = False

    requires_torch: bool = False

    requires_cuda: bool = False

    cpu_supported: bool = True

    gpu_supported: bool = False

    checkpoint_available: bool

    checkpoint_path: str | None = None

    checkpoint_compatibility: dict[str, Any] = Field(
        description=(
            "compatible / checked / mismatches / note"
        ),
    )

    supported_metrics: list[str] = Field(
        min_length=1,
    )

    supports_uncertainty: bool

    compute_cost: str = Field(
        description=(
            "trivial | seconds | minutes | hours"
        ),
    )

    status: str = Field(
        min_length=1,
        description=(
            "benchmark_active | checkpoint_ready | "
            "legacy_review_only | needs_validation | "
            "needs_retraining"
        ),
    )

    family: str | None = None

    capability_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Scientific mechanism capabilities such as "
            "temporal_dependency, physics_constraint, "
            "feature_selection, optimization_surrogate."
        ),
    )

    source: str | None = None

    source_code_path: str | None = None

    source_available: bool | None = None
    train_from_source_supported: bool | None = None
    checkpoint_inference_supported: bool | None = None
    architecture_factory: str | None = None
    adapter_available: bool = False
    runner_callable: bool = False
    required_dependencies: list[str] = Field(default_factory=list)
    data_contract_compatible: bool = True
    remaining_blocker: str | None = None

    @property
    def task_list(self) -> list[str]:
        return (
            self.task_types
            or self.supported_tasks
        )

    @property
    def is_trainable(self) -> bool:
        if self.trainable is not None:
            return self.trainable

        return self.training_available

    @property
    def is_inference_supported(self) -> bool:
        if (
            self.inference_supported
            is not None
        ):
            return self.inference_supported

        return self.inference_available

    @property
    def has_source(self) -> bool:
        return self.source_available if self.source_available is not None else bool(self.source_code_path)

    @property
    def can_train_from_source(self) -> bool:
        if self.train_from_source_supported is not None:
            return self.train_from_source_supported
        return self.has_source and self.is_trainable

    @property
    def can_infer_from_checkpoint(self) -> bool:
        if self.checkpoint_inference_supported is not None:
            return self.checkpoint_inference_supported
        return self.is_inference_supported and self.checkpoint_available


class ModelRegistry:
    """
    Unified registry of all available soft-sensing small models.

    Planner / Runner only see this unified interface;
    they never touch sklearn / torch internals directly.
    """

    def __init__(
        self,
        specs: Iterable[ModelSpec] | None = None,
    ):
        self._models: dict[str, ModelSpec] = {}

        for spec in specs or []:
            self.register(spec)

    # ---------------------------------------------------------
    # Registration
    # ---------------------------------------------------------

    def register(self, spec: ModelSpec) -> None:
        key = spec.model_name.strip().lower()

        if not key:
            raise ValueError(
                "model_name_required"
            )

        if key in self._models:
            raise ValueError(
                f"duplicate_model_registration:{key}"
            )

        self._models[key] = spec

    def register_many(
        self,
        specs: Iterable[ModelSpec],
    ) -> None:
        for spec in specs:
            self.register(spec)

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(self, model_name: str) -> ModelSpec:
        key = model_name.strip().lower()

        if key not in self._models:
            raise KeyError(
                f"unknown_model:{key};"
                f"known={','.join(sorted(self._models))}"
            )

        return self._models[key]

    def names(self) -> list[str]:
        return sorted(self._models)

    def resolve_target(self, target_variable: str) -> str | None:
        """Resolve a target without silently replacing its scientific meaning."""
        requested = str(target_variable).strip()
        if not requested:
            return None

        targets = {
            target
            for spec in self._models.values()
            for target in spec.supported_targets
        }
        by_casefold = {target.casefold(): target for target in targets}
        exact = by_casefold.get(requested.casefold())
        if exact:
            return exact

        lowered = requested.casefold()
        if "体积" in requested or "volumetric" in lowered or "volume" in lowered:
            return by_casefold.get("steam_volumetric_flow")
        if "质量" in requested or "mass" in lowered:
            return by_casefold.get("main_steam_mass_flow")
        if "氮氧" in requested or "nox" in lowered:
            return by_casefold.get("nox")
        if "给煤" in requested or "送风" in requested:
            return by_casefold.get("optimization_objective")
        return None

    def match_task_capability(
        self,
        *,
        task_type: str,
        target_variable: str,
        metrics: list[str] | tuple[str, ...] | None = None,
        capability: ExperimentCapabilityRegistry | None = None,
    ) -> list[ModelSpec]:
        """Match registered implementations to a scientific task category."""
        task = str(task_type).strip().casefold()
        task_aliases = {
            "prediction": {
                "prediction", "soft_sensor_prediction", "nox_prediction",
                "mass_flow_forecast", "steam_volume_forecast",
            },
            "diagnosis": {"diagnosis", "fault_diagnosis"},
            "optimization": {"optimization", "optimization_surrogate"},
        }
        if task not in task_aliases:
            raise ValueError(f"unsupported_task_type:{task_type}")

        target = self.resolve_target(target_variable)
        if target is None:
            return []

        required_metrics = {
            str(item).strip().upper()
            for item in (metrics or [])
            if str(item).strip()
        }
        models = []
        for spec in self._models.values():
            if not (set(spec.task_list) & task_aliases[task]):
                continue
            if target not in spec.supported_targets:
                continue
            if required_metrics and not (
                required_metrics & {item.upper() for item in spec.supported_metrics}
            ):
                continue
            models.append(spec)

        if capability is not None:
            if target not in set(capability.available_target_variables()):
                return []
            executable = set(capability.available_models())
            executable.add(capability.reference_model_id())
            models = [
                spec for spec in models
                if spec.model_name.casefold() in executable
            ]

        return sorted(models, key=lambda spec: spec.model_name)

    def list_models(
        self,
        *,
        framework: str | None = None,
        status: str | None = None,
        task: str | None = None,
        family: str | None = None,
    ) -> list[ModelSpec]:

        result = []

        for spec in self._models.values():
            if (
                framework
                and spec.framework != framework
            ):
                continue

            if status and spec.status != status:
                continue

            if (
                task
                and task not in spec.supported_tasks
            ):
                continue

            if (
                family
                and spec.family != family
            ):
                continue

            result.append(spec)

        return sorted(
            result,
            key=lambda spec: spec.model_name,
        )

    # ---------------------------------------------------------
    # Scientific suitability filtering
    # ---------------------------------------------------------

    def filter_compatible(
        self,
        *,
        tasks: (
            list[str] | tuple[str, ...] | None
        ) = None,
        target: str | None = None,
        features: int | None = None,
        window_steps: int | None = None,
        horizon_steps: int | None = None,
        sampling_interval_seconds: int | None = None,
        metrics: (
            list[str] | tuple[str, ...] | None
        ) = None,
        inference_required: bool = False,
        training_required: bool = False,
        checkpoint_compatible_required: bool = False,
        statuses: (
            list[str] | tuple[str, ...] | None
        ) = None,
    ) -> list[ModelSpec]:
        """
        Scientifically suitable models for a required contract.

        This does NOT decide current executability; that is
        the job of ExperimentCapabilityRegistry.
        """

        required_tasks = {
            str(item).strip()
            for item in (tasks or [])
            if str(item).strip()
        }

        required_metrics = {
            str(item).strip().upper()
            for item in (metrics or [])
            if str(item).strip()
        }

        allowed_statuses = {
            str(item).strip()
            for item in (statuses or [])
            if str(item).strip()
        }

        result = []

        for spec in self._models.values():

            if (
                required_tasks
                and not (
                    required_tasks
                    & set(spec.supported_tasks)
                )
            ):
                continue

            if (
                target
                and target not in spec.supported_targets
            ):
                continue

            if (
                features is not None
                and spec.required_features != features
            ):
                continue

            window_steps_required = (
                spec.window_requirements.get("steps")
            )

            if (
                window_steps is not None
                and window_steps_required is not None
                and window_steps_required != window_steps
            ):
                continue

            sampling_required = (
                spec.window_requirements.get(
                    "sampling_interval_seconds"
                )
            )

            if (
                sampling_interval_seconds is not None
                and sampling_required is not None
                and sampling_required
                != sampling_interval_seconds
            ):
                continue

            if horizon_steps is not None:
                supported = spec.horizon_capability.get(
                    "supported_steps",
                    [],
                )

                horizon_steps_required = (
                    spec.horizon_capability.get("steps")
                )

                matches = (
                    horizon_steps_required
                    == horizon_steps
                    or horizon_steps in supported
                )

                if not matches:
                    continue

            if (
                required_metrics
                and not (
                    required_metrics
                    & set(spec.supported_metrics)
                )
            ):
                continue

            if (
                inference_required
                and not spec.inference_available
            ):
                continue

            if (
                training_required
                and not spec.training_available
            ):
                continue

            if (
                checkpoint_compatible_required
                and spec.checkpoint_available
                and not spec.checkpoint_compatibility.get(
                    "compatible",
                    False,
                )
            ):
                continue

            if (
                allowed_statuses
                and spec.status not in allowed_statuses
            ):
                continue

            result.append(spec)

        return sorted(
            result,
            key=lambda spec: spec.model_name,
        )

    # ---------------------------------------------------------
    # Executability intersection (CapabilityRegistry)
    # ---------------------------------------------------------

    def compatible_with_capability(
        self,
        capability: ExperimentCapabilityRegistry,
        *,
        tasks: (
            list[str] | tuple[str, ...] | None
        ) = None,
        target: str | None = None,
        metrics: (
            list[str] | tuple[str, ...] | None
        ) = None,
        window_steps: int | None = None,
        horizon_steps: int | None = None,
        sampling_interval_seconds: int | None = None,
    ) -> list[ModelSpec]:
        """
        Scientifically suitable AND currently executable.

        The CapabilityRegistry is the ONLY authority on
        current executability; the ModelRegistry is the
        catalog of what exists.
        """

        suitable = self.filter_compatible(
            tasks=tasks,
            target=target,
            metrics=metrics,
            window_steps=(
                window_steps
                or capability.window_steps
            ),
            horizon_steps=(
                horizon_steps
                or capability.prediction_horizon_steps
            ),
            sampling_interval_seconds=(
                sampling_interval_seconds
                or capability.sampling_interval_seconds
            ),
        )

        executable_names = set(
            capability.available_models()
        )

        executable_names.add(
            capability.reference_model_id()
        )

        return [
            spec
            for spec in suitable
            if spec.model_name.lower()
            in executable_names
        ]

    # ---------------------------------------------------------
    # Adapter factory
    # ---------------------------------------------------------

    def build_adapter(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Return the unified fit/predict/evaluate adapter.

        Adapter construction is lazy: heavy imports happen
        inside the adapter, not at registry import time.
        """

        from boilermind.models.adapters import (
            build_adapter_for_spec,
        )

        spec = self.get(model_name)

        return build_adapter_for_spec(
            spec,
            **kwargs,
        )
