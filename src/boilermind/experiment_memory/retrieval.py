from __future__ import annotations

import re
from collections import Counter
from typing import Any

from boilermind.core.contracts import (
    ComparisonLevel,
    EvidenceTier,
    ExperimentMemoryBundle,
    ExperimentMemoryHit,
    ExperimentObservation,
    HistoricalExperimentRecord,
    ObservationType,
    ResearchProblemSpec,
)

from .comparison import scope_from_problem
from .store import ExperimentMemoryStore


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower()))


def _level(query, record: HistoricalExperimentRecord) -> tuple[ComparisonLevel, list[str]]:
    reasons: list[str] = []
    hard_fields = ("target_variable", "prediction_mode", "thermodynamic_standard", "prediction_horizon_steps")
    hard_mismatch = []
    known_match = []
    unknown = []
    for name in hard_fields:
        left, right = getattr(query, name), getattr(record.scope, name)
        if left in (None, "") or right in (None, ""):
            unknown.append(name)
        elif str(left).lower() == str(right).lower():
            known_match.append(name)
        else:
            hard_mismatch.append(name)
    if hard_mismatch:
        return ComparisonLevel.NOT_COMPARABLE, [f"hard_scope_mismatch:{name}" for name in hard_mismatch]
    if query.dataset_sha256 and record.scope.dataset_sha256 and query.dataset_sha256 == record.scope.dataset_sha256:
        secondary = ("window_steps", "split_policy", "regime_definition")
        mismatched = [name for name in secondary if getattr(query, name) not in (None, "") and getattr(record.scope, name) not in (None, "") and str(getattr(query, name)).lower() != str(getattr(record.scope, name)).lower()]
        if not mismatched and not unknown:
            return ComparisonLevel.DIRECTLY_COMPARABLE, ["same_dataset_hash", *[f"scope_match:{name}" for name in known_match]]
        return ComparisonLevel.CONDITIONALLY_COMPARABLE, [
            "same_dataset_hash",
            *[f"conditional_mismatch:{name}" for name in mismatched],
            *[f"scope_unknown:{name}" for name in unknown],
        ]
    if len(known_match) >= 3:
        return ComparisonLevel.CONDITIONALLY_COMPARABLE, [*[f"scope_match:{name}" for name in known_match], "dataset_identity_not_confirmed"]
    if known_match:
        return ComparisonLevel.TRANSFER_ONLY, [*[f"scope_match:{name}" for name in known_match], *[f"scope_unknown:{name}" for name in unknown]]
    return ComparisonLevel.UNKNOWN, ["insufficient_scope_identity"]


def retrieve_experiment_memory(
    problem: ResearchProblemSpec | dict[str, Any],
    capability: dict[str, Any],
    store: ExperimentMemoryStore,
    *,
    top_k: int = 12,
) -> ExperimentMemoryBundle:
    problem_model = problem if isinstance(problem, ResearchProblemSpec) else ResearchProblemSpec.model_validate(problem)
    query_scope = scope_from_problem(problem_model, capability)
    query_tokens = _tokens(" ".join((problem_model.original_question, problem_model.research_object, problem_model.target_variable, problem_model.operating_condition, problem_model.research_goal)))
    records = store.load_records()
    observations = store.load_observations()
    by_experiment: dict[str, list[ExperimentObservation]] = {}
    for observation in observations:
        for experiment_id in observation.source_experiment_ids:
            by_experiment.setdefault(experiment_id, []).append(observation)

    hits: list[ExperimentMemoryHit] = []
    for record in records:
        if record.evidence_tier == EvidenceTier.PLANNED_NOT_EXECUTED:
            continue
        level, reasons = _level(query_scope, record)
        if level in {ComparisonLevel.NOT_COMPARABLE, ComparisonLevel.UNKNOWN}:
            continue
        text = " ".join((record.raw_context, record.raw_hypothesis, record.raw_result, record.raw_limitations))
        lexical = len(query_tokens & _tokens(text)) / max(len(query_tokens), 1)
        scope_bonus = {
            ComparisonLevel.DIRECTLY_COMPARABLE: 0.7,
            ComparisonLevel.CONDITIONALLY_COMPARABLE: 0.45,
            ComparisonLevel.TRANSFER_ONLY: 0.2,
        }[level]
        tier_bonus = {
            EvidenceTier.AUDITED_CONFIRMATORY: 0.2,
            EvidenceTier.AUDITED_EXPLORATORY: 0.12,
            EvidenceTier.LEGACY_INFORMATIVE: 0.05,
            EvidenceTier.ENGINEERING_FAILURE: 0.0,
        }.get(record.evidence_tier, 0.0)
        if record.evidence_tier == EvidenceTier.AUDITED_CONFIRMATORY and record.audit_status != "PASSED":
            tier_bonus = 0.06
            reasons.append("confirmatory_claim_requires_source_artifact_verification")
        hits.append(ExperimentMemoryHit(
            experiment_id=record.experiment_id,
            observation_ids=[item.observation_id for item in by_experiment.get(record.experiment_id, [])],
            comparison_level=level,
            retrieval_score=min(1.0, scope_bonus + tier_bonus + lexical * 0.1),
            retrieval_reasons=[*reasons, f"lexical_overlap:{lexical:.3f}", f"evidence_tier:{record.evidence_tier.value}"],
        ))
    hits.sort(key=lambda item: (-item.retrieval_score, item.experiment_id))
    hits = hits[:top_k]
    selected_observation_ids = {oid for hit in hits for oid in hit.observation_ids}
    selected = [item for item in observations if item.observation_id in selected_observation_ids]
    valid = [item for item in selected if not item.invalid_for_scientific_synthesis]
    planned = sorted({record.hypothesis_id for record in records if record.evidence_tier == EvidenceTier.PLANNED_NOT_EXECUTED and record.hypothesis_id})
    return ExperimentMemoryBundle(
        problem_id=problem_model.problem_id,
        directly_comparable=[item for item in hits if item.comparison_level == ComparisonLevel.DIRECTLY_COMPARABLE],
        conditionally_comparable=[item for item in hits if item.comparison_level == ComparisonLevel.CONDITIONALLY_COMPARABLE],
        transfer_only=[item for item in hits if item.comparison_level == ComparisonLevel.TRANSFER_ONLY],
        supported_observations=[item for item in valid if item.observation_type == ObservationType.SUPPORTED],
        falsified_observations=[item for item in valid if item.observation_type == ObservationType.FALSIFIED],
        contradictions=[item for item in valid if item.observation_type in {ObservationType.CONTRADICTION, ObservationType.BOUNDARY_CONDITION}],
        engineering_failures=[item for item in selected if item.observation_type in {ObservationType.ENGINEERING_FAILURE, ObservationType.DATA_QUALITY_WARNING}],
        completed_experiment_ids=sorted({hit.experiment_id for hit in hits}),
        planned_hypothesis_ids=planned,
        retrieval_notes=[
            "structured_scope_filter_applied_before_lexical_scoring",
            "record_count_does_not_increase_scientific_priority",
            "planned_and_invalid_records_excluded_from_scientific_support",
        ],
    )
