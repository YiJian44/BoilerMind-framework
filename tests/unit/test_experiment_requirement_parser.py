import pytest

from boilermind.planning.experiment_requirement_parser import (
    EXPERIMENT_TYPE_CONSTRAINED_OPTIMIZATION,
    EXPERIMENT_TYPE_FEATURE_INTERVENTION,
    EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON,
    EXPERIMENT_TYPE_DIRECTION_RATE_INTERACTION,
    EXPERIMENT_TYPE_SENSOR_CORRUPTION,
    parse_hypothesis_requirements,
    freeze_hypothesis_design,
    frozen_design_alignment_issues,
)


def make_direction_rate_hypothesis():
    return {
        "id": "H-RATE",
        "hypothesis": (
            "深度调峰下，降负荷过程中误差随负荷变化速率增加的"
            "幅度大于升负荷过程。"
        ),
        "verification_intent": (
            "比较升负荷和降负荷方向下MAE与变化速率的关系。"
        ),
        "falsification_condition": (
            "降负荷误差斜率小于或等于升负荷误差斜率。"
        ),
        "variables": ["负荷变化速率", "升降负荷方向", "MAE"],
    }


def make_h001():
    """
    lag feature vs no-lag feature hypothesis.
    """

    return {
        "id": "H001",
        "hypothesis_id": "H001",
        "hypothesis": (
            "在锅炉深度调峰运行工况下，为ridge、"
            "bayesianridge和hgb模型引入时间序列滞后特征后，"
            "其在真实数据上的10分钟后主蒸汽流量预测MAE与"
            "RMSE将低于未使用滞后特征的对应基线模型。"
        ),
        "verification_intent": (
            "执行model_comparison与chronological_validation"
            "操作，在相同训练/测试划分下，对比各模型有/无"
            "滞后特征输入时在锁定测试集上的MAE与RMSE。"
        ),
        "falsification_condition": (
            "在至少一个启用模型上，引入滞后特征后其MAE与"
            "RMSE均未降低，且该结果在chronological_validation"
            "协议下稳定复现。"
        ),
        "variables": [
            "时间序列滞后特征（存在 vs. 不存在）",
            "ridge模型预测MAE",
        ],
    }


def make_h002():
    """
    reference_model_comparison hypothesis:
    ridge/bayesianridge/hgb vs persistence.
    """

    return {
        "id": "H002",
        "hypothesis_id": "H002",
        "hypothesis": (
            "在真实锅炉深度调峰运行数据上，ridge、"
            "bayesianridge与hgb模型对10分钟后主蒸汽流量"
            "的预测MAE与RMSE将高于persistence模型。"
        ),
        "verification_intent": (
            "执行reference_model_comparison操作，在锁定"
            "测试集上计算各启用模型与persistence模型的MAE"
            "与RMSE，并进行数值比较。"
        ),
        "falsification_condition": (
            "至少一个启用模型（ridge/bayesianridge/hgb）"
            "在MAE与RMSE两个指标上同时优于persistence模型，"
            "且该优势在chronological_validation下保持一致。"
        ),
        "variables": [
            "ridge模型预测MAE",
            "persistence模型预测MAE",
        ],
    }


def make_future_constrained_problem():
    """
    Future multi-variable + constraint problem that must be
    recognized as NOT currently executable.
    """

    return {
        "id": "H090",
        "hypothesis_id": "H090",
        "hypothesis": (
            "同时调整给煤量与给水流量，在未来10分钟主汽"
            "压力不超过23 MPa的约束下，最大化蒸汽体积流量"
            "预测精度。"
        ),
        "verification_intent": (
            "执行multi_variable_intervention与"
            "multi_target_forecast，评估给煤量+给水流量"
            "干预对蒸汽体积流量的影响。"
        ),
        "falsification_condition": (
            "在压力约束下无法找到比当前输入更优的多变量"
            "干预组合。"
        ),
    }


def test_h001_parses_as_feature_intervention():
    requirements = parse_hypothesis_requirements(
        make_h001()
    )

    assert requirements.hypothesis_id == "H001"
    assert (
        requirements.experiment_type
        == EXPERIMENT_TYPE_FEATURE_INTERVENTION
    )

    assert "feature_intervention" in (
        requirements.required_operations
    )
    assert "model_comparison" in (
        requirements.required_operations
    )
    assert "chronological_validation" in (
        requirements.required_operations
    )
    assert "locked_test_evaluation" in (
        requirements.required_operations
    )

    assert (
        requirements.requires_feature_intervention
        is True
    )

    assert set(
        requirements.required_models
    ) == {
        "ridge",
        "bayesianridge",
        "hgb",
    }

    assert set(
        requirements.required_metrics
    ) == {
        "MAE",
        "RMSE",
    }


def test_h002_parses_as_reference_model_comparison():
    requirements = parse_hypothesis_requirements(
        make_h002()
    )

    assert requirements.hypothesis_id == "H002"
    assert (
        requirements.experiment_type
        == EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON
    )

    assert "reference_model_comparison" in (
        requirements.required_operations
    )
    assert "locked_test_evaluation" in (
        requirements.required_operations
    )
    assert "chronological_validation" in (
        requirements.required_operations
    )

    assert set(
        requirements.required_models
    ) == {
        "ridge",
        "bayesianridge",
        "hgb",
        "persistence",
    }

    assert (
        requirements.required_model_roles[
            "persistence"
        ]
        == "reference"
    )

    assert set(
        requirements.required_metrics
    ) == {
        "MAE",
        "RMSE",
    }

    assert (
        requirements.prediction_horizon_steps
        == 40
    )

    assert (
        requirements.required_targets
        == ["main_steam_mass_flow"]
    )


def test_future_constrained_problem_fails_closed_in_parser():
    requirements = parse_hypothesis_requirements(
        make_future_constrained_problem()
    )

    assert (
        requirements.experiment_type
        == EXPERIMENT_TYPE_CONSTRAINED_OPTIMIZATION
    )

    for operation in [
        "multi_variable_intervention",
        "multi_target_forecast",
        "hard_constraint_evaluation",
        "constrained_optimization",
    ]:
        assert operation in (
            requirements.required_operations
        )

    assert (
        "steam_volumetric_flow"
        in requirements.required_targets
    )

    assert requirements.hard_constraints


def test_parser_requires_hypothesis_id():
    hypothesis = make_h002()
    hypothesis.pop("id")
    hypothesis.pop("hypothesis_id")

    with pytest.raises(
        ValueError,
        match="hypothesis_id_required",
    ):
        parse_hypothesis_requirements(
            hypothesis
        )


def test_parser_requires_statement():
    hypothesis = make_h002()
    hypothesis.pop("hypothesis")

    with pytest.raises(
        ValueError,
        match="hypothesis_statement_required",
    ):
        parse_hypothesis_requirements(
            hypothesis
        )


@pytest.mark.parametrize("label,expected", [("h80", 80), ("H 40", 40)])
def test_explicit_horizon_labels_are_parsed(label, expected):
    hypothesis = make_h002()
    hypothesis["hypothesis"] = f"比较 Ridge 与 Transformer 的 {label} 预测"
    requirements = parse_hypothesis_requirements(hypothesis)
    assert requirements.prediction_horizon_steps == expected


def test_pairwise_model_comparison_compiles_to_deterministic_criteria():
    hypothesis = make_h002()
    hypothesis["hypothesis"] = (
        "Transformer和LSTM相比Ridge并不能降低MAE"
    )
    requirements = parse_hypothesis_requirements(hypothesis)
    assert requirements.confirmation_criteria == [
        "all_models_not_better_than_model_on:lstm,transformer|ridge|MAE,RMSE"
    ]
    assert requirements.falsification_criteria == [
        "any_model_better_than_model_on:lstm,transformer|ridge|MAE,RMSE"
    ]


def test_direction_rate_hypothesis_never_falls_back_to_locked_test():
    requirements = parse_hypothesis_requirements(
        make_direction_rate_hypothesis()
    )
    assert (
        requirements.experiment_type
        == EXPERIMENT_TYPE_DIRECTION_RATE_INTERACTION
    )
    assert set(requirements.required_operations) >= {
        "load_rate_computation",
        "direction_regime_assignment",
        "rate_stratified_evaluation",
        "direction_rate_interaction_evaluation",
    }
    assert "locked_test_evaluation" not in requirements.required_operations


def test_frozen_design_is_stable_until_hypothesis_semantics_change():
    hypothesis = make_direction_rate_hypothesis()
    hypothesis["scientific_design"] = freeze_hypothesis_design(
        hypothesis
    ).model_dump(mode="json")

    assert frozen_design_alignment_issues(hypothesis) == []

    hypothesis["hypothesis"] += "并额外注入高斯噪声。"

    issues = frozen_design_alignment_issues(hypothesis)
    assert "hypothesis_design:semantic_drift:experiment_type" in issues
    assert "hypothesis_design:semantic_drift:required_operations" in issues


def test_lower_error_relation_keeps_left_model_as_candidate():
    requirements = parse_hypothesis_requirements({
        "id": "H-LOWER",
        "hypothesis": "Ridge模型预测MAE和RMSE低于Persistence模型。",
        "verification_intent": "在锁定测试集比较Ridge与Persistence。",
        "falsification_condition": "Ridge不能同时降低MAE和RMSE。",
        "variables": [
            "prediction_horizon_steps", "steam_volumetric_flow", "Ridge",
            "Persistence", "MAE", "RMSE",
        ],
    })
    assert requirements.confirmation_criteria == [
        "all_models_better_than_model_on:ridge|persistence|MAE,RMSE"
    ]
    assert requirements.falsification_criteria == [
        "any_model_not_better_than_model_on:ridge|persistence|MAE,RMSE"
    ]
    assert requirements.required_variables == []
    assert "single_reference_relation_required" not in (
        requirements.required_operations
    )


def test_multi_reference_claim_is_not_partially_compiled():
    requirements = parse_hypothesis_requirements({
        "id": "H-MULTI-REF",
        "hypothesis": (
            "Ridge的MAE和RMSE低于BayesianRidge与Persistence。"
        ),
        "verification_intent": "比较三种模型。",
        "falsification_condition": "任一关系不成立。",
    })
    assert "single_reference_relation_required" in (
        requirements.required_operations
    )


def test_noise_robustness_hypothesis_compiles_required_interventions():
    requirements = parse_hypothesis_requirements({
        "id": "H-NOISE",
        "hypothesis": (
            "随着传感器噪声水平增加，LSTM相对Ridge的MAE优势减小。"
        ),
        "verification_intent": (
            "注入不同强度的高斯噪声与毛刺，比较误差退化曲线。"
        ),
        "falsification_condition": (
            "LSTM与Ridge的误差退化斜率没有差异。"
        ),
        "variables": ["sensor_noise_level", "MAE"],
    })
    assert requirements.experiment_type == EXPERIMENT_TYPE_SENSOR_CORRUPTION
    assert set(requirements.required_operations) >= {
        "sensor_corruption_injection",
        "gaussian_noise_injection",
        "spike_injection",
        "corruption_level_sweep",
        "clean_corrupted_paired_comparison",
        "robustness_degradation_evaluation",
    }
    assert "locked_test_evaluation" not in requirements.required_operations
