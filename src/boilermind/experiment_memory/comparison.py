from __future__ import annotations

import re
from typing import Any

from boilermind.core.contracts import (
    ComparisonLevel,
    ExperimentComparison,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
    ResearchProblemSpec,
)


_HORIZON_PATTERN = re.compile(r"(?:@|h(?:orizon)?\s*[=:]?\s*)(\d+)", re.I)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def scope_from_problem(
    problem: ResearchProblemSpec | dict[str, Any],
    capability: dict[str, Any] | None = None,
) -> ExperimentScopeSignature:
    payload = problem.model_dump() if isinstance(problem, ResearchProblemSpec) else dict(problem)
    capability = capability or {}
    question = " ".join(str(payload.get(key, "")) for key in (
        "original_question", "target_variable", "operating_condition", "research_goal"
    ))
    lowered = question.lower()
    horizon = None
    match = _HORIZON_PATTERN.search(lowered)
    if match:
        horizon = int(match.group(1))
    elif "20分钟" in question or "20 min" in lowered:
        horizon = 80
    elif "10分钟" in question or "10 min" in lowered:
        horizon = 40

    if "direct-v" in lowered or "体积" in question or "volum" in lowered:
        prediction_mode = "direct_volume"
    elif "质量" in question or re.search(r"\bm\s*@", lowered):
        prediction_mode = "mass"
    else:
        prediction_mode = None

    thermodynamic = "IF97" if "if97" in lowered else None
    dataset = capability.get("dataset_contract") or capability.get("dataset") or {}
    return ExperimentScopeSignature(
        target_variable=str(payload.get("target_variable") or "") or None,
        target_definition=str(payload.get("research_goal") or "") or None,
        prediction_mode=prediction_mode,
        thermodynamic_standard=thermodynamic,
        dataset_id=dataset.get("dataset_id") or dataset.get("id"),
        dataset_sha256=dataset.get("dataset_hash") or dataset.get("sha256"),
        window_steps=capability.get("window_steps") or dataset.get("window_steps"),
        prediction_horizon_steps=horizon or capability.get("prediction_horizon_steps"),
        sampling_interval_seconds=capability.get("sampling_interval_seconds"),
        regime_definition=str(payload.get("operating_condition") or "") or None,
        metrics=list(capability.get("available_metrics") or capability.get("metrics") or []),
        baselines=[str(capability.get("reference_model"))] if capability.get("reference_model") else [],
    )


def compare_experiment_scopes(
    left: HistoricalExperimentRecord,
    right: HistoricalExperimentRecord,
) -> ExperimentComparison:
    fields = (
        "target_variable", "target_unit", "prediction_mode", "thermodynamic_standard",
        "dataset_sha256", "feature_set_id", "feature_count", "window_steps",
        "prediction_horizon_steps", "sampling_interval_seconds", "split_policy",
        "regime_definition",
    )
    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    for field in fields:
        lval = getattr(left.scope, field)
        rval = getattr(right.scope, field)
        if lval in (None, "", []) or rval in (None, "", []):
            unknown.append(field)
        elif _norm(lval) == _norm(rval):
            matched.append(field)
        else:
            mismatched.append(field)

    hard = {"target_variable", "target_unit", "prediction_mode", "thermodynamic_standard", "prediction_horizon_steps"}
    if hard.intersection(mismatched):
        level = ComparisonLevel.NOT_COMPARABLE
    elif "dataset_sha256" in matched and not mismatched:
        level = ComparisonLevel.DIRECTLY_COMPARABLE
    elif not hard.intersection(unknown) and set(mismatched).issubset({"dataset_sha256", "split_policy", "regime_definition"}):
        level = ComparisonLevel.CONDITIONALLY_COMPARABLE
    elif matched and not hard.intersection(mismatched):
        level = ComparisonLevel.TRANSFER_ONLY
    else:
        level = ComparisonLevel.UNKNOWN
    return ExperimentComparison(
        left_experiment_id=left.experiment_id,
        right_experiment_id=right.experiment_id,
        level=level,
        matched_fields=matched,
        mismatched_fields=mismatched,
        unknown_fields=unknown,
        rationale=f"matched={matched}; mismatched={mismatched}; unknown={unknown}",
    )
