"""Adapt core experiment contracts to the independent evolution graph format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .kg_update import DEFAULT_GRAPH_PATH, update_evolution_graph


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TypeError(f"{name} must be a mapping or provide model_dump()")


def _text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip()


def _required_text(payload: Mapping[str, Any], field: str, owner: str) -> str:
    value = _text(payload.get(field, ""))
    if not value:
        raise ValueError(f"{owner}.{field} must be non-empty")
    return value


def _scope_text(hypothesis: Mapping[str, Any]) -> str:
    scope = hypothesis.get(
        "applicable_scope",
        hypothesis.get(
            "applicability_conditions",
            hypothesis.get("condition", ""),
        ),
    )
    if isinstance(scope, (list, tuple)):
        return "；".join(_text(item) for item in scope if _text(item))
    return _text(scope)


def adapt_experiment_feedback(
    experiment_result: Any,
    scientific_result: Any,
    hypothesis: Any,
) -> dict[str, Any]:
    """Convert real experiment/scientific contracts to ``kg_update`` input."""
    experiment = _as_dict(experiment_result, "experiment_result")
    scientific = _as_dict(scientific_result, "scientific_result")
    hypothesis_info = _as_dict(hypothesis, "hypothesis")

    experiment_id = _required_text(experiment, "experiment_id", "experiment_result")
    hypothesis_id = _required_text(experiment, "hypothesis_id", "experiment_result")
    if _required_text(scientific, "experiment_id", "scientific_result") != experiment_id:
        raise ValueError("experiment_id mismatch between experiment and scientific results")
    if _required_text(scientific, "hypothesis_id", "scientific_result") != hypothesis_id:
        raise ValueError("hypothesis_id mismatch between experiment and scientific results")

    declared_hypothesis_id = _required_text(hypothesis_info, "hypothesis_id", "hypothesis")
    if declared_hypothesis_id != hypothesis_id:
        raise ValueError("hypothesis_id mismatch between result and hypothesis")

    content = _text(
        hypothesis_info.get(
            "hypothesis",
            hypothesis_info.get(
                "hypothesis_statement",
                hypothesis_info.get("content", ""),
            ),
        )
    )
    if not content:
        raise ValueError("hypothesis content must be non-empty")

    mechanism_chain = _text(
        hypothesis_info.get(
            "mechanism_chain",
            hypothesis_info.get(
                "engineering_mechanism",
                hypothesis_info.get("mechanism", ""),
            ),
        )
    )
    if not mechanism_chain:
        raise ValueError("hypothesis.mechanism_chain must be non-empty")

    experiment_valid = experiment.get("experiment_valid")
    if not isinstance(experiment_valid, bool):
        raise ValueError("experiment_result.experiment_valid must be a boolean")

    metrics = experiment.get("metrics", {})
    model_records = experiment.get("model_records", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("experiment_result.metrics must be an object")
    if not isinstance(model_records, Mapping):
        raise ValueError("experiment_result.model_records must be an object")

    return {
        "experiment_id": experiment_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis": content,
        "mechanism_chain": mechanism_chain,
        "verdict": _required_text(scientific, "verdict", "scientific_result").lower(),
        "metrics": dict(metrics),
        "experiment_valid": experiment_valid,
        "status": _required_text(experiment, "status", "experiment_result").lower(),
        "model_records": dict(model_records),
        "applicable_scope": _scope_text(hypothesis_info),
    }


def update_evolution_from_results(
    experiment_result: Any,
    scientific_result: Any,
    hypothesis: Any,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Adapt real results and persist the resulting graph update."""
    payload = adapt_experiment_feedback(
        experiment_result,
        scientific_result,
        hypothesis,
    )
    return update_evolution_graph(payload, graph_path)
