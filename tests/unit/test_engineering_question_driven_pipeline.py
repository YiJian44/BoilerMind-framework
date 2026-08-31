from boilermind.experiment_memory.hypothesis_assessment import (
    assess_hypotheses_with_memory,
)
from boilermind.orchestration.problem_intake import analyze_problem_intake
from boilermind.orchestration.qwen_problem_parser import QwenProblemParser
from boilermind.skills.ranking_skill import RankingSkill


def _observation(observation_id, observation_type, claim, *, target="steam_volumetric_flow", horizon=80):
    return {
        "observation_id": observation_id,
        "source_experiment_ids": [f"EXP-{observation_id}"],
        "observation_type": observation_type,
        "claim": claim,
        "scope_signature": {
            "target_variable": target,
            "prediction_horizon_steps": horizon,
        },
        "supporting_metrics": {},
        "counter_evidence": [],
        "confidence_level": 0.8,
        "reuse_policy": "same_scope_only",
    }


def test_natural_engineering_question_reaches_hypothesis_generation_without_model_details():
    decision = analyze_problem_intake(
        "为什么深度调峰升负荷时，主蒸汽流量估计误差往往比降负荷时大？"
    )
    assert decision.status == "READY_FOR_HYPOTHESIS"
    assert "candidate_models" not in decision.missing_fields
    assert "metrics" not in decision.missing_fields
    assert "prediction_horizon" not in decision.missing_fields


def test_qwen_target_normalization_removes_estimation_error_suffix():
    payload = {"target_variable": "main steam flow estimation error"}
    QwenProblemParser._separate_target_and_objective(
        payload,
        "为什么主蒸汽流量估计误差在升负荷时更大？",
    )
    assert payload["target_variable"] == "main_steam_mass_flow"


def test_history_is_attached_after_generation_without_changing_raw_hypothesis():
    hypothesis = {
        "hypothesis_id": "H001",
        "title": "方向响应差异",
        "hypothesis_statement": "升负荷阶段的蒸汽流量估计误差高于降负荷阶段",
        "engineering_mechanism": "升负荷过程存在更强的动态滞后",
        "expected_observation": "升负荷分层MAE高于降负荷分层MAE",
        "key_variables": ["负荷方向", "MAE"],
        "applicability_conditions": ["深度调峰"],
        "falsification_condition": "升负荷MAE不高于降负荷MAE",
        "evidence_needed": ["方向分层预测误差"],
        "raw_hypothesis_sha256": "a" * 64,
    }
    memory = {
        "problem_id": "P-1",
        "supported_observations": [
            _observation("OBS-S", "SUPPORTED", "升负荷阶段蒸汽流量MAE高于降负荷阶段")
        ],
        "falsified_observations": [],
        "contradictions": [
            _observation("OBS-C", "CONTRADICTION", "降负荷阶段蒸汽流量MAE高于升负荷阶段")
        ],
        "engineering_failures": [],
    }
    result = assess_hypotheses_with_memory(
        [hypothesis], memory,
        {"target_variable": "steam_volumetric_flow", "required_horizon_steps": 80},
    )[0]
    assert result["hypothesis_statement"] == hypothesis["hypothesis_statement"]
    assert result["raw_hypothesis_sha256"] == "a" * 64
    assert result["historical_assessment"]["historical_support_level"] == "MIXED"
    assert result["historical_assessment"]["conflicting_observations"] == ["OBS-C"]


def test_scientific_rank_keeps_unexecutable_engineering_hypothesis():
    hypothesis = {
        "hypothesis_id": "H001",
        "title": "故障漂移",
        "hypothesis": "表计漂移时冗余热力变量能够降低估计误差累积",
        "hypothesis_statement": "表计漂移时冗余热力变量能够降低估计误差累积",
        "mechanism": "冗余变量提供独立信息",
        "engineering_mechanism": "冗余变量提供独立信息",
        "expected_observation": "漂移注入后冗余估计的误差增幅更小",
        "verification_intent": "执行漂移注入并比较误差增幅",
        "falsification_condition": "冗余估计的误差增幅不更小",
        "historical_assessment": {
            "duplicate_status": "NEW", "conflicting_observations": []
        },
    }
    output = RankingSkill().execute({
        "qualified_hypotheses": [hypothesis],
        "research_problem": {
            "original_question": "主蒸汽流量表漂移时能否依靠其他测点稳定估计？",
            "required_objective_dimensions": ["fault_robustness"],
        },
        "scientific_context": {
            "supported_experiment_operations": ["model_comparison"],
            "enabled_experiment_models": ["ridge"],
            "available_metrics": ["MAE"],
        },
    })
    assert len(output["ranking"]) == 1
    mapping = output["qualified_hypotheses"][0]["verification_mapping"]
    assert mapping["recommended_action"] == "NEEDS_NEW_OPERATION"
    assert mapping["executable_now"] is False
    assert output["selected_hypothesis_id"] is None


def test_direction_question_is_not_rescued_by_narrower_observable_premise():
    hypothesis = {
        "hypothesis_id": "H001",
        "hypothesis": "升负荷蓄热造成动态估计滞后",
        "hypothesis_statement": "升负荷蓄热造成动态估计滞后",
        "engineering_mechanism": "蓄热动态非对称",
        "expected_observation": "升负荷MAE高于降负荷MAE且与金属温度相关",
        "falsification_condition": "升负荷MAE不高于降负荷MAE",
        "historical_assessment": {"duplicate_status": "NEW"},
    }
    output = RankingSkill().execute({
        "qualified_hypotheses": [hypothesis],
        "research_problem": {
            "original_question": "为什么深度调峰升负荷时，主蒸汽流量估计误差往往比降负荷时大？",
        },
        "scientific_context": {
            "supported_experiment_operations": ["regime_stratified_evaluation"],
            "enabled_experiment_models": ["ridge"],
            "available_metrics": ["MAE"],
        },
    })
    mapping = output["qualified_hypotheses"][0]["verification_mapping"]
    assert output["selected_hypothesis_id"] is None
    assert mapping["executable_now"] is False
    assert mapping["recommended_action"] == "NEEDS_NEW_OPERATION"
    assert "verification_scope" not in mapping
