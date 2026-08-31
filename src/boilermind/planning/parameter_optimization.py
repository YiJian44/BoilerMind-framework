from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

from boilermind.experiment.metric_normalizer import (
    canonical_metric_name,
    numeric_normalized_metrics,
)


class ExperimentOptimizationResult(BaseModel):
    optimized_variable: str
    candidates: list[int | float | str]
    candidate_results: list[dict[str, Any]] = Field(default_factory=list)
    best_candidate: int | float | str | None = None
    selection_metric: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    status: str


def expand_parameter_plans(
    base_plan: dict[str, Any],
    *,
    variable: str,
    candidates: list[int | float | str],
) -> list[dict[str, Any]]:
    if variable != "window_steps":
        raise ValueError(f"unsupported_optimization_variable:{variable}")
    plans: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, int):
            raise ValueError("window_steps_candidate_must_be_integer")
        plan = deepcopy(base_plan)
        plan[variable] = candidate
        plan["plan_id"] = f"PLAN-window-{candidate}"
        plan["experiment_type"] = "model_comparison"
        plans.append(plan)
    return plans


def collect_parameter_candidate_results(
    round_plans: list[dict[str, Any]],
    round_outcomes: list[dict[str, Any]],
    *,
    variable: str,
    selection_metric: str,
) -> list[dict[str, Any]]:
    """Adapt completed experiment rounds for validation-only comparison."""

    metric = canonical_metric_name(selection_metric)
    candidate_results: list[dict[str, Any]] = []
    for round_plan, round_outcome in zip(
        round_plans, round_outcomes, strict=True
    ):
        result = round_outcome["experiment_result"]
        comparable_records: list[tuple[Any, dict[str, float]]] = []
        for record in result.model_records.values():
            validation_metrics = dict(record.validation_metrics)
            validation_metrics.update(
                numeric_normalized_metrics(record.validation_metrics)
            )
            if (
                record.fit_success
                and isinstance(validation_metrics.get(metric), (int, float))
            ):
                comparable_records.append((record, validation_metrics))
        selector = max if metric == "R2" else min
        selected_pair = selector(
            comparable_records,
            key=lambda pair: pair[1][metric],
            default=None,
        )
        selected_record = selected_pair[0] if selected_pair else None
        candidate_results.append({
            "candidate": round_plan.get(variable),
            "experiment_id": result.experiment_id,
            "experiment_valid": bool(round_outcome["audit"].execution_valid),
            "validation_metrics": selected_pair[1] if selected_pair else {},
            "selected_model": (
                selected_record.model_name if selected_record is not None else None
            ),
            # Locked-test metrics are retained for reporting only. The selector
            # above and compare_parameter_results never read this field.
            "locked_test_metrics": dict(result.metrics),
        })
    return candidate_results


def compare_parameter_results(
    candidate_results: list[dict[str, Any]],
    *,
    variable: str,
    candidates: list[int | float | str],
    selection_metric: str = "MAE",
) -> ExperimentOptimizationResult:
    """Select from validation metrics only; locked test never chooses a candidate."""
    comparable: list[tuple[float, dict[str, Any]]] = []
    metric = canonical_metric_name(selection_metric)
    for item in candidate_results:
        if item.get("experiment_valid") is False:
            continue
        metrics = dict(
            item.get("validation_metrics")
            or item.get("normalized_metrics")
            or {}
        )
        metrics.update(numeric_normalized_metrics(metrics))
        value = metrics.get(metric)
        if isinstance(value, (int, float)):
            comparable.append((float(value), item))
    if not comparable:
        return ExperimentOptimizationResult(
            optimized_variable=variable,
            candidates=candidates,
            candidate_results=candidate_results,
            selection_metric=metric,
            confidence=0.0,
            reason=f"no_validation_{metric}_available",
            status="INSUFFICIENT_VALIDATION_EVIDENCE",
        )
    reverse = metric == "R2"
    comparable.sort(key=lambda pair: pair[0], reverse=reverse)
    best_value, best = comparable[0]
    confidence = 0.0
    if len(comparable) > 1:
        second = comparable[1][0]
        denominator = max(abs(second), 1e-12)
        confidence = min(1.0, abs(second - best_value) / denominator)
    return ExperimentOptimizationResult(
        optimized_variable=variable,
        candidates=candidates,
        candidate_results=candidate_results,
        best_candidate=best.get("candidate"),
        selection_metric=metric,
        confidence=confidence,
        reason=f"best_candidate_selected_by_validation_{metric}",
        status="PASS",
    )
