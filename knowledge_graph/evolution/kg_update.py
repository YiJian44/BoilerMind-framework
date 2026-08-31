"""Persist experiment feedback to the independent evolution graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "evolution_graph.json"
_REQUIRED_FIELDS = (
    "experiment_id",
    "hypothesis_id",
    "hypothesis",
    "mechanism_chain",
    "verdict",
    "metrics",
    "experiment_valid",
)
_VALID_VERDICTS = {
    "supported",
    "falsified",
    "inconclusive",
    "insufficient_evidence",
    "partially_supported",
}


def _load_graph(graph_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not graph_path.exists():
        return {"nodes": [], "edges": []}
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("evolution graph must contain nodes and edges lists")
    return graph


def _upsert_by_id(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    for index, existing in enumerate(items):
        if existing.get("id") == item["id"]:
            items[index] = item
            return
    items.append(item)


def _upsert_edge(edges: list[dict[str, Any]], edge: dict[str, Any]) -> None:
    identity = (edge["source"], edge["type"], edge["target"])
    for index, existing in enumerate(edges):
        if (existing.get("source"), existing.get("type"), existing.get("target")) == identity:
            edges[index] = edge
            return
    edges.append(edge)


def _validate_result(experiment_result: Mapping[str, Any]) -> None:
    missing = [field for field in _REQUIRED_FIELDS if field not in experiment_result]
    if missing:
        raise ValueError(f"experiment_result missing required fields: {', '.join(missing)}")
    for field in (
        "experiment_id",
        "hypothesis_id",
        "hypothesis",
        "mechanism_chain",
        "verdict",
    ):
        if not str(experiment_result[field]).strip():
            raise ValueError(f"experiment_result.{field} must be non-empty")
    if not isinstance(experiment_result["metrics"], Mapping):
        raise ValueError("experiment_result.metrics must be an object")
    if not isinstance(experiment_result["experiment_valid"], bool):
        raise ValueError("experiment_result.experiment_valid must be a boolean")
    verdict = str(experiment_result["verdict"]).strip().lower()
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"unsupported experiment_result.verdict: {verdict}")


def update_evolution_graph(
    experiment_result: Mapping[str, Any],
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Add hypothesis, result, and validated hypothesis to an evolution graph.

    Nodes and edges are upserted by deterministic identifiers, making repeated
    delivery of the same experiment result idempotent.
    """
    _validate_result(experiment_result)
    path = Path(graph_path)
    graph = _load_graph(path)

    hypothesis_id = str(experiment_result["hypothesis_id"]).strip()
    experiment_id = str(experiment_result["experiment_id"]).strip()
    verdict = str(experiment_result["verdict"]).strip().lower()
    experiment_valid = experiment_result["experiment_valid"]
    experiment_status = str(experiment_result.get("status", "")).strip().lower()
    execution_completed = not experiment_status or experiment_status == "completed"
    applicable_scope = str(
        experiment_result.get(
            "applicable_scope",
            experiment_result.get("condition", ""),
        )
    ).strip()
    validation_succeeded = (
        experiment_valid
        and execution_completed
        and verdict == "supported"
    )
    has_previous_validation = any(
        node.get("type") == "ValidatedHypothesis"
        and node.get("hypothesis_id") == hypothesis_id
        for node in graph["nodes"]
    )

    hypothesis_node = {
        "id": hypothesis_id,
        "type": "Hypothesis",
        "content": str(experiment_result["hypothesis"]).strip(),
        "status": (
            "validated"
            if validation_succeeded or has_previous_validation
            else "invalid_experiment"
            if not experiment_valid
            else "experiment_not_completed"
            if not execution_completed
            else verdict
        ),
    }
    result_node = {
        "id": experiment_id,
        "type": "ExperimentResult",
        "verdict": verdict,
        "metrics": dict(experiment_result["metrics"]),
        "experiment_valid": experiment_valid,
    }
    if experiment_status:
        result_node["status"] = experiment_status
    if "model_records" in experiment_result:
        result_node["model_records"] = dict(experiment_result["model_records"])
    _upsert_by_id(graph["nodes"], hypothesis_node)
    _upsert_by_id(graph["nodes"], result_node)
    _upsert_edge(
        graph["edges"],
        {"source": hypothesis_id, "type": "validated_by", "target": experiment_id},
    )

    if validation_succeeded:
        validated_id = f"VH-{hypothesis_id}-{experiment_id}"
        validated_node = {
            "id": validated_id,
            "type": "ValidatedHypothesis",
            "hypothesis_id": hypothesis_id,
            "content": str(experiment_result["hypothesis"]).strip(),
            "mechanism_chain": str(
                experiment_result["mechanism_chain"]
            ).strip(),
            "validation_status": "supported",
            "experiment_id": experiment_id,
            "metrics": dict(experiment_result["metrics"]),
            "applicable_scope": applicable_scope,
        }
        _upsert_by_id(graph["nodes"], validated_node)
        _upsert_edge(
            graph["edges"],
            {"source": experiment_id, "type": "generates", "target": validated_id},
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return graph
