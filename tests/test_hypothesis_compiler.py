from __future__ import annotations

from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
    ExperimentCapabilityRegistry,
)
from boilermind.hypothesis.hypothesis_compiler import (
    classify_claims,
    compile_hypotheses,
)
from boilermind.models.execution_environment import ExecutionEnvironment
from boilermind.skills.planning_skill import PlanningSkill
from boilermind.skills.ranking_skill import RankingSkill
from boilermind.skills.hypothesis_skill import HypothesisGenerationSkill


def _problem() -> dict:
    return {
        "problem_id": "P-COMPILE",
        "original_question": "不同负荷变化状态对未来目标预测误差的影响",
        "target_variable": "steam_volumetric_flow",
        "operating_condition": "深度调峰升负荷工况",
        "metrics": ["MAE", "RMSE", "R2"],
        "required_horizon_steps": 40,
        "required_objective_dimensions": [],
    }


def _context(*operations: str) -> dict:
    return {
        "supported_experiment_operations": list(operations),
        "available_metrics": ["MAE", "RMSE", "R2", "MBE"],
        "available_variables": ["steam_volumetric_flow"],
        "supported_prediction_horizon_steps": [40],
        "enabled_experiment_models": ["ridge", "bayesianridge"],
        "reference_model": "persistence",
    }


def _hypothesis(statement: str, *, hypothesis_id: str = "H001") -> dict:
    return {
        "id": hypothesis_id,
        "hypothesis_id": hypothesis_id,
        "title": "科研假设",
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "mechanism": "可观测预测误差随运行状态改变。",
        "engineering_mechanism": "可观测预测误差随运行状态改变。",
        "inference": "比较各预声明状态的MAE。",
        "expected_observation": "比较各预声明状态的MAE。",
        "verification_intent": "比较ramp_up、ramp_down与steady工况的MAE。",
        "falsification_condition": "ramp_up的MAE未高于ramp_down的MAE。",
        "confirmation_criteria": ["predeclared_target_achieved"],
        "falsification_criteria": ["predeclared_target_not_achieved"],
        "variables": ["steam_volumetric_flow"],
        "key_variables": ["steam_volumetric_flow"],
        "trigger_types": ["HUMAN_PROPOSAL"],
        "duplicate_check": {},
        "historical_assessment": {},
    }


def test_fully_supported_hypothesis_remains_unchanged() -> None:
    hypothesis = _hypothesis(
        "在locked_test上评价steam_volumetric_flow预测MAE。"
    )
    hypothesis["verification_intent"] = "执行locked_test_evaluation并计算MAE。"
    hypothesis["falsification_condition"] = "locked_test评价未产生MAE。"
    compiled, records = compile_hypotheses(
        [hypothesis], _problem(), _context("locked_test_evaluation")
    )

    assert compiled[0]["hypothesis_id"] == "H001"
    assert compiled[0]["hypothesis"] == hypothesis["hypothesis"]
    assert records[0].adaptation_reason == "hypothesis_fully_supported_unchanged"


def test_model_comparison_question_freezes_one_registered_model_hypothesis() -> None:
    problem = _problem()
    problem["original_question"] = "在调峰工况下，哪种模型预测蒸汽体积流量效果最好？"
    context = _context(
        "model_comparison", "reference_model_comparison",
        "chronological_validation", "locked_test_evaluation",
    )
    context["locked_test_supported"] = True
    context["prediction_horizon_steps"] = 40
    compiled, records = compile_hypotheses(
        [_hypothesis("ramp_up与ramp_down误差不同。", hypothesis_id="H001"),
         _hypothesis("不同模型可能适用不同工况。", hypothesis_id="H002")],
        problem,
        context,
    )
    assert len(compiled) == 1
    candidate = compiled[0]
    assert candidate["hypothesis_id"] == "H001"
    assert candidate["scientific_design"]["experiment_type"] == "reference_model_comparison"
    assert candidate["scientific_design"]["required_models"] == [
        "ridge", "bayesianridge", "persistence",
    ]
    assert "unsupported_scientific_design" not in candidate[
        "scientific_design"
    ]["required_operations"]
    assert candidate["scientific_design"]["prediction_horizon_steps"] == 40
    assert records[0].current_executable is True
    ranking = RankingSkill().execute({
        "qualified_hypotheses": compiled,
        "research_problem": problem,
        "scientific_context": context,
    })
    assert ranking["selected_hypothesis_id"] == "H001"
    assert ranking["qualified_hypotheses"][0]["verification_mapping"][
        "executable_now"
    ] is True


def test_explicit_model_comparison_scope_is_not_expanded_to_registry() -> None:
    problem = _problem()
    problem.update({
        "original_question": "比较Ridge、RandomForest与Persistence的预测效果",
        "required_operations": ["model_comparison", "reference_model_comparison"],
        "required_models": ["ridge", "rf"],
        "reference_models": ["persistence"],
    })
    context = _context(
        "model_comparison", "reference_model_comparison",
        "chronological_validation", "locked_test_evaluation",
    )
    context["enabled_experiment_models"] = [
        "ridge", "bayesianridge", "rf", "transformer",
    ]
    context["locked_test_supported"] = True
    compiled, _records = compile_hypotheses(
        [_hypothesis("比较指定模型的预测误差。")], problem, context
    )

    design = compiled[0]["scientific_design"]
    assert design["required_models"] == ["ridge", "rf", "persistence"]
    assert design["treatment"]["models"] == ["ridge", "rf"]
    assert "transformer" not in design["required_models"]


def test_numeric_percentage_claim_is_downgraded() -> None:
    hypothesis = _hypothesis(
        "升负荷状态导致steam_volumetric_flow预测误差增加5%，并比较ramp_up与ramp_down。"
    )
    compiled, records = compile_hypotheses(
        [hypothesis], _problem(), _context("regime_stratified_evaluation")
    )

    assert compiled[0]["hypothesis_id"] == "H001-A"
    assert "5%" not in compiled[0]["hypothesis"]
    assert any(item.claim_type == "numeric_claim" for item in records[0].removed_claims)


def test_causal_claim_is_downgraded() -> None:
    hypothesis = _hypothesis(
        "升负荷状态导致预测误差放大，并比较ramp_up与ramp_down。"
    )
    _compiled, records = compile_hypotheses(
        [hypothesis], _problem(), _context("regime_stratified_evaluation")
    )
    assert any(item.claim_type == "causal_claim" for item in records[0].removed_claims)


def test_statistical_claim_is_downgraded() -> None:
    hypothesis = _hypothesis(
        "ramp_up与ramp_down的预测误差存在统计显著差异。"
    )
    _compiled, records = compile_hypotheses(
        [hypothesis], _problem(), _context("regime_stratified_evaluation")
    )
    assert any(item.claim_type == "statistical_claim" for item in records[0].removed_claims)


def test_claim_classifier_reports_unsupported_operation() -> None:
    claims = classify_claims(
        _hypothesis("ramp_up与ramp_down误差不同。"),
        ["operation:causal_analysis"],
    )
    assert any(
        item.claim_type == "unsupported_operation_claim"
        and item.text == "causal_analysis"
        for item in claims
    )


def test_compiled_narrow_variant_does_not_rescue_original_hypothesis() -> None:
    environment = ExecutionEnvironment(
        os="test",
        python_version="3.11",
        sklearn_available=True,
        torch_available=False,
        cuda_available=False,
        gpu_available=False,
    )
    capability = DirectVolume31VCapabilityRegistry(environment=environment)
    context = capability.to_scientific_context()
    context.update(_context(*context["supported_experiment_operations"]))
    original = _hypothesis(
        "升负荷速率导致预测误差增加5%且达到统计显著，并比较ramp_up与ramp_down。"
    )
    compiled, _records = compile_hypotheses([original], _problem(), context)
    ranking = RankingSkill().execute({
        "qualified_hypotheses": compiled,
        "research_problem": _problem(),
        "scientific_context": context,
    })

    assert ranking["selected_hypothesis_id"] is None
    assert ranking["status"] == "blocked_no_comprehensive_candidate"


def test_fully_unsupported_claim_remains_fail_closed() -> None:
    original = _hypothesis("未知微观效应需要causal_analysis才能验证。")
    compiled, records = compile_hypotheses(
        [original], _problem(), _context("locked_test_evaluation")
    )
    assert compiled[0]["compilation_status"] == "UNSUPPORTED"
    assert records[0].current_executable is False


def test_generation_compilation_cannot_bypass_hard_gate(
    monkeypatch,
) -> None:
    environment = ExecutionEnvironment(
        os="test", python_version="3.11", sklearn_available=True,
        torch_available=False, cuda_available=False, gpu_available=False,
    )
    capability = DirectVolume31VCapabilityRegistry(environment=environment)
    scientific_context = capability.to_scientific_context()
    scientific_context["available_variables"] = ["steam_volumetric_flow"]
    scientific_context["supported_prediction_horizon_steps"] = [40, 80]
    problem = _problem()
    problem["original_question"] = (
        "深度调峰升负荷工况下，不同负荷变化状态对未来10分钟"
        "蒸汽体积流量预测影响"
    )
    raw = {
        "title": "升负荷速率效应",
        "hypothesis_statement": (
            "升负荷速率导致预测误差增加5%，且ramp_up与ramp_down"
            "差异达到统计显著。"
        ),
        "engineering_mechanism": "负荷变化导致预测误差变化。",
        "expected_observation": "ramp_up误差显著高于ramp_down。",
        "key_variables": ["steam_volumetric_flow"],
        "applicability_conditions": ["深度调峰升负荷工况"],
        "falsification_condition": "ramp_up误差未高于ramp_down。",
        "assumptions": [],
        "evidence_needed": [],
    }
    generation = HypothesisGenerationSkill()
    monkeypatch.setattr(generation, "_generate_seeds", lambda *_args: [raw])
    generated = generation.execute({
        "research_problem": problem,
        "scientific_context": scientific_context,
        "hypothesis_generation_mode": "fast",
    })
    assert generated["qualified_hypotheses"][0]["hypothesis_id"] == "H001-A"

    ranking = RankingSkill().execute({
        **generated,
        "research_problem": problem,
        "scientific_context": scientific_context,
    })
    assert ranking["selected_hypothesis_id"] is None
    assert ranking["status"] == "blocked_no_comprehensive_candidate"
