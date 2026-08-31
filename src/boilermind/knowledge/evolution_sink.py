from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Protocol

from boilermind.core.contracts import (
    ExperimentResult,
    ScientificHypothesis,
    ScientificResult,
)


class ExperimentEvolutionSink(Protocol):
    """Optional post-verdict sink for experiment-driven knowledge growth."""

    def record(
        self,
        experiment_result: ExperimentResult,
        scientific_result: ScientificResult,
        hypothesis: ScientificHypothesis | Mapping[str, Any],
    ) -> None:
        ...


class JsonEvolutionSink:
    """Persist validated hypotheses through the independent JSON evolution module."""

    def __init__(self, graph_path: str | Path | None = None):
        self.graph_path = (
            Path(graph_path)
            if graph_path is not None
            else Path(__file__).resolve().parents[3]
            / "knowledge_graph"
            / "evolution"
            / "evolution_graph.json"
        )

    def record(
        self,
        experiment_result: ExperimentResult,
        scientific_result: ScientificResult,
        hypothesis: ScientificHypothesis | Mapping[str, Any],
    ) -> None:
        # Lazy import keeps the canonical ResearchOrchestrator independent from the
        # repository-level knowledge_graph implementation.
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from knowledge_graph.evolution.experiment_adapter import (
            update_evolution_from_results,
        )

        update_evolution_from_results(
            experiment_result,
            scientific_result,
            hypothesis,
            graph_path=self.graph_path,
        )


def adapt_full_pipeline_hypothesis(
    hypothesis: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt FullPipeline's generated hypothesis without inventing data."""
    hypothesis_id = str(
        hypothesis.get("hypothesis_id") or hypothesis.get("id") or ""
    ).strip()
    content = str(hypothesis.get("hypothesis") or "").strip()
    mechanism_chain = str(
        hypothesis.get("mechanism_chain") or hypothesis.get("mechanism") or ""
    ).strip()

    missing = [
        name
        for name, value in (
            ("hypothesis_id", hypothesis_id),
            ("hypothesis", content),
            ("mechanism_chain", mechanism_chain),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "full_pipeline_evolution_hypothesis_missing:" + ",".join(missing)
        )

    applicability = hypothesis.get("applicability_conditions")
    if applicability is None:
        applicability = hypothesis.get("conditions", [])

    return {
        "hypothesis_id": hypothesis_id,
        "hypothesis": content,
        "mechanism_chain": mechanism_chain,
        "applicability_conditions": applicability,
    }
