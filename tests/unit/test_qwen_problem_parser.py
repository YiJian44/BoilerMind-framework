import pytest

from boilermind.orchestration.qwen_problem_parser import (
    QwenProblemParser,
    QwenProblemParserError,
)


def _valid_payload():
    return {
        "research_object": "boiler",
        "target_variable": "operation stability",
        "objective": "improve prediction quality",
        "metrics": ["MAE", "RMSE"],
        "operating_condition": "load change",
        "manipulated_variables": [
            "coal feed",
            "air supply",
        ],
        "observed_variables": [],
        "context_variables": [],
        "research_goal": (
            "Study coordinated boiler operation."
        ),
        "success_criteria": [
            "The question can be experimentally tested."
        ],
        "constraints": [],
    }


def test_single_string_list_field_is_normalized():
    payload = _valid_payload()

    payload["success_criteria"] = (
        "The question can be experimentally tested."
    )

    QwenProblemParser._validate_payload(
        payload
    )

    assert payload["success_criteria"] == [
        "The question can be experimentally tested."
    ]


def test_invalid_list_field_still_fails_closed():
    payload = _valid_payload()

    payload["success_criteria"] = {
        "criterion": "invalid structure"
    }

    with pytest.raises(
        QwenProblemParserError,
        match="success_criteria must be a list",
    ):
        QwenProblemParser._validate_payload(
            payload
        )


def test_prediction_accuracy_is_not_used_as_target_variable():
    payload = _valid_payload()
    payload["target_variable"] = "预测精度"

    QwenProblemParser._validate_payload(payload)
    QwenProblemParser._separate_target_and_objective(
        payload,
        "增加历史实验窗口数，是否可以改善调峰工况下软测量模型预测精度",
    )

    assert payload["target_variable"] == "unspecified"
    assert payload["objective"] == "improve_prediction_accuracy"
    assert payload["metrics"] == ["MAE", "RMSE", "R2"]


def test_h80_is_horizon_and_direct_v_resolves_physical_target():
    payload = _valid_payload()
    payload["target_variable"] = "h80"

    QwenProblemParser._separate_target_and_objective(
        payload,
        "31变量直接V预测任务中验证h80跨时间块稳定性",
    )

    assert payload["target_variable"] == "steam_volumetric_flow"
    assert "horizon_label_removed_from_target" in payload["target_inference_reason"]


def test_horizon_label_without_physical_target_fails_closed():
    payload = _valid_payload()
    payload["target_variable"] = "horizon=80"

    QwenProblemParser._separate_target_and_objective(
        payload,
        "验证horizon=80的稳定性",
    )

    assert payload["target_variable"] == "unspecified"


def test_natural_language_volume_target_is_canonicalized():
    payload = _valid_payload()
    payload["target_variable"] = "steam volumetric flow"

    QwenProblemParser._separate_target_and_objective(
        payload,
        "direct-V h80 prediction",
    )

    assert payload["target_variable"] == "steam_volumetric_flow"


def test_descriptive_direct_steam_volume_target_is_canonicalized():
    payload = _valid_payload()
    payload["target_variable"] = "direct steam volumetric flow"

    QwenProblemParser._separate_target_and_objective(payload, "direct-V h80")

    assert payload["target_variable"] == "steam_volumetric_flow"


def test_chinese_direct_steam_volume_alias_is_canonicalized():
    payload = _valid_payload()
    payload["target_variable"] = "直接蒸汽体积流量"
    QwenProblemParser._separate_target_and_objective(payload, "预测直接蒸汽体积流量")
    assert payload["target_variable"] == "steam_volumetric_flow"


def test_chinese_steam_flow_alias_is_canonicalized_as_mass_flow():
    payload = _valid_payload()
    payload["target_variable"] = "蒸汽流量"
    QwenProblemParser._separate_target_and_objective(payload, "估计蒸汽量")
    assert payload["target_variable"] == "main_steam_mass_flow"


def test_generic_english_steam_flow_alias_is_canonicalized_as_mass_flow():
    payload = _valid_payload()
    payload["target_variable"] = "Steam flow"
    QwenProblemParser._separate_target_and_objective(payload, "estimate steam flow")
    assert payload["target_variable"] == "main_steam_mass_flow"


def test_user_declared_execution_constraints_are_deterministic():
    parsed = QwenProblemParser._execution_constraints(
        "比较 Persistence、Ridge、Bayesian Ridge、随机森林、LSTM和Transformer"
        "的未来20分钟(h80)预测；必须validation选模，locked-test仅最终评估"
    )
    assert parsed["required_models"] == [
        "bayesianridge", "ridge", "rf", "lstm", "transformer",
    ]
    assert parsed["reference_models"] == ["persistence"]
    assert parsed["required_horizon_steps"] == 80
    assert set(parsed["required_operations"]) == {
        "model_comparison", "chronological_validation", "locked_test_evaluation",
    }


def test_multi_objective_dimensions_are_not_delegated_to_qwen():
    assert QwenProblemParser._execution_constraints(
        "既要估得准，仪表故障时不能乱，还要低算力部署"
    )["required_models"] == []
