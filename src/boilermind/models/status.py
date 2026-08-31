from __future__ import annotations

from pathlib import Path
from typing import Any

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.models.catalog import (
    build_default_registry,
)

from boilermind.models.execution_environment import (
    ExecutionEnvironment,
)

from boilermind.models.adapters import (
    build_adapter_for_spec,
)


ADAPTER_STATUS_RUNNER_CALLABLE = "RUNNER_CALLABLE"

ADAPTER_STATUS_SOURCE_ONLY = "SOURCE_ONLY"

ADAPTER_STATUS_MISSING_DEPENDENCY = (
    "MISSING_DEPENDENCY"
)

ADAPTER_STATUS_CHECKPOINT_INCOMPATIBLE = (
    "CHECKPOINT_INCOMPATIBLE"
)

ADAPTER_STATUS_NEEDS_ADAPTER = "NEEDS_ADAPTER"

ADAPTER_STATUS_IMPLEMENTATION_BLOCKED = (
    "IMPLEMENTATION_BLOCKED"
)


def model_status_matrix(
    *,
    model_registry: Any | None = None,
    capability: ExperimentCapabilityRegistry | None = None,
    environment: ExecutionEnvironment | None = None,
) -> list[dict[str, Any]]:
    """
    Per-model readiness report:

      REGISTERED             in ModelRegistry
      EXECUTABLE             in current CapabilityRegistry pool
                             (or reference model)
      TRAINABLE              training_available
      CHECKPOINT_COMPATIBLE  checkpoint available AND compatible
      RUNNER_CALLABLE        can the Runner instantiate + execute
                             it in the CURRENT environment
    """

    registry = (
        model_registry or build_default_registry()
    )

    capability = (
        capability or ExperimentCapabilityRegistry()
    )

    environment = (
        environment or ExecutionEnvironment.detect()
    )

    executable_names = set(
        capability.available_models()
    )

    executable_names.add(
        capability.reference_model_id()
    )

    rows: list[dict[str, Any]] = []

    for spec in registry.list_models():
        executable = (
            spec.model_name in executable_names
        )

        adapter_available = _adapter_constructible(
            spec
        )

        source_available = (
            spec.source_code_path is not None
            or spec.framework
            in {"sklearn", "heuristic"}
        )

        checkpoint_compatible = (
            spec.checkpoint_compatibility.get(
                "compatible",
                False,
            )
            if spec.checkpoint_available
            else None
        )

        adapter_status, reason = (
            _adapter_status(
                spec,
                executable,
                environment,
            )
        )

        runner_callable = bool(executable and (spec.framework in {"sklearn", "heuristic"} or spec.runner_callable))

        rows.append(
            {
                "model_name": spec.model_name,
                "framework": spec.framework,
                "status": spec.status,
                "REGISTERED": True,
                "EXECUTABLE": executable,
                "TRAINABLE": spec.is_trainable,
                "CHECKPOINT_COMPATIBLE": (
                    checkpoint_compatible
                ),
                "RUNNER_CALLABLE": runner_callable,
                "adapter_status": adapter_status,
                "reason": reason,
                "adapter_available": (
                    adapter_available
                ),
                "source_available": source_available,
                "SOURCE_AVAILABLE": spec.has_source,
                "TRAIN_FROM_SOURCE_SUPPORTED": spec.can_train_from_source,
                "CHECKPOINT_AVAILABLE": spec.checkpoint_available,
                "CHECKPOINT_INFERENCE_SUPPORTED": spec.can_infer_from_checkpoint,
                "ADAPTER_AVAILABLE": spec.adapter_available or adapter_available,
                "CURRENTLY_EXECUTABLE": executable,
                "checkpoint_required": (
                    spec.checkpoint_required
                ),
                "source_code_path": (
                    spec.source_code_path
                ),
                "source_code_exists": (
                    (
                        Path(__file__).resolve().parent
                        / spec.source_code_path
                    ).is_file()
                    if spec.source_code_path
                    else None
                ),
            }
        )

    return rows


def _adapter_constructible(spec) -> bool:
    try:
        build_adapter_for_spec(spec)
        return True
    except Exception:
        return False


def _adapter_status(
    spec,
    executable: bool,
    environment: ExecutionEnvironment,
) -> tuple[str, str]:
    """
    Classify one model's adapter readiness.

    RUNNER_CALLABLE / SOURCE_ONLY / MISSING_DEPENDENCY /
    CHECKPOINT_INCOMPATIBLE / NEEDS_ADAPTER.
    """

    if executable and (
        spec.model_name == "persistence"
        or spec.framework == "sklearn"
    ):
        return (
            ADAPTER_STATUS_RUNNER_CALLABLE,
            "",
        )

    if spec.framework == "xgboost":
        if not environment.optional_dependencies.get(
            "xgboost",
            False,
        ):
            return (
                ADAPTER_STATUS_MISSING_DEPENDENCY,
                "xgboost_not_installed",
            )

        return (
            ADAPTER_STATUS_NEEDS_ADAPTER,
            "xgboost_adapter_not_wired",
        )

    if spec.requires_torch:
        if not environment.torch_available:
            return (
                ADAPTER_STATUS_MISSING_DEPENDENCY,
                "torch_required:torch_not_installed",
            )

        if not spec.source_code_path:
            return (
                ADAPTER_STATUS_NEEDS_ADAPTER,
                "source_code_missing",
            )

        if executable and spec.adapter_available and spec.runner_callable:
            return ADAPTER_STATUS_RUNNER_CALLABLE, ""
        return ADAPTER_STATUS_SOURCE_ONLY, spec.remaining_blocker or "adapter_not_ready"

    if (
        spec.checkpoint_available
        and not spec.checkpoint_compatibility.get(
            "compatible",
            False,
        )
    ):
        return (
            ADAPTER_STATUS_CHECKPOINT_INCOMPATIBLE,
            "checkpoint_incompatible:"
            + ",".join(
                spec.checkpoint_compatibility.get(
                    "mismatches",
                    [],
                )
            ),
        )

    if not spec.source_code_path:
        return (
            ADAPTER_STATUS_NEEDS_ADAPTER,
            "source_code_missing",
        )

    return (
        ADAPTER_STATUS_SOURCE_ONLY,
        "adapter_executor_not_wired",
    )
