from __future__ import annotations

from typing import Any


EXPERIMENT_TRIGGERS = {
    "HISTORICAL_EXPERIMENT",
    "CURRENT_DATA_OBSERVATION",
    "CONTRADICTORY_RESULTS",
}
BOUNDARY_TRIGGERS = {"CAPABILITY_EXPANSION"}
LITERATURE_TRIGGERS = {"LITERATURE_INSPIRATION", "HUMAN_PROPOSAL"}


def source_category(candidate: dict[str, Any]) -> str:
    triggers = {str(item) for item in candidate.get("trigger_types", [])}
    if triggers & EXPERIMENT_TRIGGERS:
        return "experiment_or_data"
    if triggers & BOUNDARY_TRIGGERS:
        return "boundary_or_capability"
    return "literature_or_human"


def enforce_generation_quota(
    candidates: list[dict[str, Any]],
    quota: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quota = quota or {
        "experiment_or_data": 7,
        "boundary_or_capability": 2,
        "literature_or_human": 1,
    }
    counts = {key: 0 for key in quota}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        category = source_category(candidate)
        candidate["source_category"] = category
        if counts[category] >= quota[category]:
            rejected.append({
                "hypothesis": candidate,
                "decision": "REJECT",
                "issues": [f"source_quota_exceeded:{category}"],
                "rationale": "候选来源配额用于防止文献或单一来源重新主导假设生成。",
            })
            continue
        counts[category] += 1
        accepted.append(candidate)
    return accepted, rejected
