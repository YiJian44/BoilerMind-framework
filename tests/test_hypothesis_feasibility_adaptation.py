from __future__ import annotations

from boilermind.hypothesis.feasibility_adapter import (
    adapt_hypotheses_for_feasibility,
)
from boilermind.skills.ranking_skill import RankingSkill


def _hypothesis(hypothesis_id: str, statement: str) -> dict:
    return {
        "id": hypothesis_id,
        "hypothesis_id": hypothesis_id,
        "title": f"title-{hypothesis_id}",
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "verification_intent": "使用MAE验证该声明。",
        "expected_observation": "产生预声明的MAE结果。",
        "falsification_condition": "预声明结果未出现。",
        "confirmation_criteria": ["predeclared_target_achieved"],
        "falsification_criteria": ["predeclared_target_not_achieved"],
        "duplicate_check": {},
        "historical_assessment": {},
    }


def _problem() -> dict:
    return {
        "original_question": "比较不同负荷状态的目标预测误差",
        "target_variable": "generic_target",
        "operating_condition": "declared operating scope",
        "metrics": ["MAE"],
        "required_horizon_steps": 40,
        "required_objective_dimensions": [],
    }


def _context(*operations: str) -> dict:
    return {
        "supported_experiment_operations": list(operations),
        "available_metrics": ["MAE"],
        "available_variables": ["generic_target"],
        "supported_prediction_horizon_steps": [40],
    }


def test_fully_supported_hypothesis_is_unchanged() -> None:
    original = _hypothesis(
        "H001",
        "在locked_test上评价generic_target预测MAE。",
    )
    output, records = adapt_hypotheses_for_feasibility(
        [original], _problem(), _context("locked_test_evaluation")
    )

    assert output == [original]
    assert records == []


def test_partially_supported_variant_cannot_rescue_original_hypothesis() -> None:
    original = _hypothesis(
        "H001",
        "升负荷速率导致generic_target预测误差显著非线性放大，"
        "并比较ramp_up、ramp_down与steady工况。",
    )
    output, records = adapt_hypotheses_for_feasibility(
        [original], _problem(), _context("regime_stratified_evaluation")
    )

    assert len(output) == 2
    assert output[0] is original
    variant = output[1]
    assert variant["hypothesis_id"] == "H001-A"
    assert variant["original_hypothesis_id"] == "H001"
    assert variant["original_hypothesis"]["hypothesis"] == original["hypothesis"]
    assert variant["scientific_design"]["required_operations"] == [
        "regime_stratified_evaluation"
    ]
    assert records == [variant["feasibility_adaptation"]]
    assert any(
        "statistical_significance_evaluation" in claim
        for claim in records[0]["removed_claims"]
    )

    ranking = RankingSkill().execute({
        "qualified_hypotheses": output,
        "research_problem": _problem(),
        "scientific_context": _context("regime_stratified_evaluation"),
    })
    assert ranking["selected_hypothesis_id"] is None
    assert ranking["status"] == "blocked_no_comprehensive_candidate"
    selected = next(
        item for item in ranking["hypotheses"]
        if item["hypothesis_id"] == "H001-A"
    )
    assert selected["verification_mapping"]["executable_now"] is False
    assert "verification_scope" not in selected["verification_mapping"]


def test_fully_unsupported_hypothesis_remains_rejected() -> None:
    original = _hypothesis(
        "H999",
        "未知微观机理导致目标出现无法观测的非线性效应。",
    )
    output, records = adapt_hypotheses_for_feasibility(
        [original], _problem(), _context("regime_stratified_evaluation")
    )

    assert output == [original]
    assert records == []
    ranking = RankingSkill().execute({
        "qualified_hypotheses": output,
        "research_problem": _problem(),
        "scientific_context": _context("regime_stratified_evaluation"),
    })
    assert ranking["selected_hypothesis_id"] is None
    assert ranking["status"] == "blocked_no_comprehensive_candidate"
