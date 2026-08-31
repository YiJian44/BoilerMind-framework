from __future__ import annotations

import re
from typing import Any

from boilermind.core.contracts import ExperimentObservation, LiteratureRelation


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", str(text).lower()))


def match_post_experiment_literature(
    observations: list[ExperimentObservation],
    evidence_bundle: dict[str, Any] | None,
    *,
    minimum_overlap: float = 0.08,
) -> list[LiteratureRelation]:
    """Match literature after the experiment without claiming it predicted the result."""
    evidence = list((evidence_bundle or {}).get("evidence", []))
    relations: list[LiteratureRelation] = []
    for observation in observations:
        observation_tokens = _tokens(observation.claim)
        ranked = []
        for item in evidence:
            text = " ".join((str(item.get("title", "")), str(item.get("text", ""))))
            tokens = _tokens(text)
            overlap = len(observation_tokens & tokens) / max(len(observation_tokens), 1)
            if overlap >= minimum_overlap:
                ranked.append((overlap, item))
        for _, item in sorted(ranked, key=lambda pair: -pair[0])[:3]:
            support = str(item.get("claim_support", "")).lower()
            relationship = "CONTRADICTING" if support == "contradicting" else "METHOD_RELATED"
            relations.append(LiteratureRelation(
                document_id=str(item.get("document_id") or item.get("evidence_id")),
                observation_id=observation.observation_id,
                relationship=relationship,
                page_number=item.get("page_number"),
                chunk_id=item.get("chunk_id"),
                excerpt=str(item.get("text", ""))[:1000],
                applicability=str(item.get("applicability", "UNKNOWN")),
                metadata_verified=(
                    item.get("identity_status") == "VERIFIED"
                    and item.get("citation_eligibility") == "FORMALLY_CITABLE"
                ),
                formatted_citation=(
                    item.get("formatted_citation")
                    if item.get("identity_status") == "VERIFIED"
                    and item.get("citation_eligibility") == "FORMALLY_CITABLE"
                    else None
                ),
            ))
    return relations
