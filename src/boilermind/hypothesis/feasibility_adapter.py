from __future__ import annotations

from typing import Any

from .hypothesis_compiler import compile_hypotheses


def adapt_hypotheses_for_feasibility(
    hypotheses: list[dict[str, Any]],
    problem: dict[str, Any],
    scientific_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility facade over the pre-quality-gate compiler."""

    compiled, _records = compile_hypotheses(
        hypotheses, problem, scientific_context
    )
    original_ids = {
        str(item.get("hypothesis_id") or item.get("id") or "")
        for item in hypotheses
    }
    variants = [
        item for item in compiled
        if str(item.get("hypothesis_id") or item.get("id") or "")
        not in original_ids
    ]
    adaptation_records = [
        dict(item["feasibility_adaptation"])
        for item in variants
        if isinstance(item.get("feasibility_adaptation"), dict)
    ]
    return [*hypotheses, *variants], adaptation_records
