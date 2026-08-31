from __future__ import annotations

from typing import Any

from boilermind.core.contracts import HypothesisScore


WEIGHTS = {
    "historical_support": 0.40,
    "historical_scope_match": 0.20,
    "problem_relevance": 0.15,
    "reproducibility": 0.15,
    "falsifiability": 0.10,
}


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _historical_features(hypothesis: dict[str, Any]) -> tuple[float, float, float, list[str]]:
    assessment = hypothesis.get("historical_assessment") or {}
    direct = list(assessment.get("directly_supporting_observations") or [])
    conflicts = list(assessment.get("conflicting_observations") or [])
    conditional = list(assessment.get("conditionally_related_observations") or [])
    mismatches = list(assessment.get("scope_mismatches") or [])
    experiment_ids = list(hypothesis.get("source_experiment_ids") or [])

    if conflicts and not mismatches:
        return 0.0, 0.0, 0.0, ["historical_direct_conflict"]
    if len(direct) >= 2:
        support = 1.0
    elif direct:
        support = 0.85
    elif conditional:
        support = 0.40
    else:
        support = 0.0
    scope = 1.0 if direct and not mismatches else 0.65 if direct else 0.40 if conditional else 0.0
    reproducibility = _clamp(assessment.get("reproducibility", 1.0 if experiment_ids else 0.0))
    return support, scope, reproducibility, []


def score_hypothesis(
    hypothesis: dict[str, Any],
    *,
    problem_relevance: float | None = None,
    data_attribute_prior: float | None = None,
    data_attribute_weight: float = 0.20,
) -> HypothesisScore:
    hypothesis_id = str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or "").strip()
    if not hypothesis_id:
        raise ValueError("hypothesis_id_required_for_scoring")
    support, scope, reproducibility, dropped = _historical_features(hypothesis)
    mapping = hypothesis.get("verification_mapping") or {}
    if not mapping.get("executable_now", hypothesis.get("current_executable", False)):
        dropped.append("not_currently_executable")
    proposal_triggers = {
        str(item).strip().upper()
        for item in hypothesis.get("trigger_types") or []
    }
    executable_proposal = bool(mapping.get("executable_now")) and bool(
        proposal_triggers & {"HUMAN_PROPOSAL", "SYSTEM_DEFAULT"}
    )
    if (
        not (hypothesis.get("source_experiment_ids") or hypothesis.get("source_observation_ids"))
        and not executable_proposal
    ):
        dropped.append("no_empirical_grounding")

    relevance = _clamp(
        problem_relevance
        if problem_relevance is not None
        else hypothesis.get("problem_relevance", 1.0)
    )
    confirmation = hypothesis.get("confirmation_criteria") or []
    falsification = hypothesis.get("falsification_criteria") or hypothesis.get("falsification_condition")
    falsifiability = 1.0 if confirmation and falsification else 0.0
    if relevance == 0.0:
        dropped.append("unrelated_to_problem")
    if falsifiability == 0.0:
        dropped.append("criterion_not_computable")

    prior = sum((
        support * WEIGHTS["historical_support"],
        scope * WEIGHTS["historical_scope_match"],
        relevance * WEIGHTS["problem_relevance"],
        reproducibility * WEIGHTS["reproducibility"],
        falsifiability * WEIGHTS["falsifiability"],
    ))

    # 数据属性先验（增量）：仅当提供 data_attribute_prior 时以附加分加入，
    # 不改变现有假设（未提供者）的评分。属性分来自数据属性画像的 property_scores，
    # 为"哪个模型软测 V 更优"提供数据依据的排序信号。
    attribute_prior = _clamp(
        data_attribute_prior
        if data_attribute_prior is not None
        else hypothesis.get("data_attribute_prior")
    )
    if attribute_prior > 0.0:
        prior += data_attribute_weight * attribute_prior

    prior = round(_clamp(prior), 6)
    return HypothesisScore(
        hypothesis_id=hypothesis_id,
        historical_support=support,
        historical_scope_match=scope,
        problem_relevance=relevance,
        reproducibility=reproducibility,
        falsifiability=falsifiability,
        prior_score=prior,
        dynamic_score=prior,
        eligible=not dropped,
        dropped_reasons=list(dict.fromkeys(dropped)),
    )


def rank_hypotheses(hypotheses: list[dict[str, Any]]) -> list[HypothesisScore]:
    scored = [score_hypothesis(item) for item in hypotheses]
    return sorted(scored, key=lambda item: (-int(item.eligible), -item.dynamic_score, item.hypothesis_id))
