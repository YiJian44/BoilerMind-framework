from boilermind.hypothesis.deterministic_admission import (
    evaluate_candidate,
    required_objective_dimensions,
)
from boilermind.skills.ranking_skill import RankingSkill


def _problem():
    return {
        "original_question": "既要准确、仪表故障时鲁棒，也要低算力部署",
        "required_objective_dimensions": [
            "accuracy", "fault_robustness", "deployment_efficiency",
        ],
    }


def _capability():
    return {
        "enabled_experiment_models": ["ridge"],
        "reference_model": "persistence",
        "supported_experiment_operations": [
            "locked_test_evaluation", "model_comparison",
            "reference_model_comparison",
            "sensor_fault_injection", "fallback_evaluation",
            "runtime_benchmark", "resource_profile",
        ],
        "available_metrics": ["MAE", "runtime"],
    }


def _hypothesis(hid="H001", full=True):
    statement = "Ridge的locked-test MAE低于Persistence"
    intent = "比较MAE"
    if full:
        statement += "；故障注入时启用Persistence降级通道减少灾难性误差，并降低推理延迟"
        intent += "、故障注入与fallback，并进行runtime推理时间基准"
    return {
        "id": hid,
        "hypothesis_id": hid,
        "title": "系统方案",
        "hypothesis": statement,
        "mechanism": "输入异常时降级，正常时使用线性模型",
        "verification_intent": intent,
        "falsification_condition": "任一预声明主指标不优于对照",
        "source_observation_ids": ["OBS-1"],
        "duplicate_check": {"duplicate": False, "similarity": 0.1},
    }


def test_required_dimensions_are_extracted_locally():
    assert required_objective_dimensions(
        "估得准，仪表故障不乱，现场算力低"
    ) == ["accuracy", "fault_robustness", "deployment_efficiency"]


def test_colloquial_meter_failure_is_fault_robustness():
    assert required_objective_dimensions(
        "深调大范围变负荷，既要估得准，又怕表出毛病时乱套，还要现场跑得动"
    ) == ["accuracy", "fault_robustness", "deployment_efficiency"]


def test_candidate_coverage_and_capability_are_programmatic():
    admission = evaluate_candidate(_hypothesis(), _problem(), _capability())
    assert admission["missing_objective_dimensions"] == []
    assert admission["missing_capabilities"] == []
    assert admission["eligible_as_comprehensive_champion"] is True


def test_accuracy_only_candidate_cannot_win_comprehensive_question():
    context = {
        "research_problem": _problem(),
        "scientific_context": _capability(),
        "qualified_hypotheses": [_hypothesis(full=False)],
    }
    result = RankingSkill().execute(context)
    assert result["selected_hypothesis_id"] is None
    assert result["status"] == "blocked_no_comprehensive_candidate"
    assert result["ranking"][0]["missing_objective_dimensions"] == [
        "fault_robustness", "deployment_efficiency",
    ]


def test_unregistered_operation_is_a_local_capability_blocker():
    capability = _capability()
    capability["supported_experiment_operations"] = ["locked_test_evaluation"]
    admission = evaluate_candidate(_hypothesis(), _problem(), capability)
    assert "operation:sensor_fault_injection" in admission["missing_capabilities"]
    assert admission["current_executable"] is False
