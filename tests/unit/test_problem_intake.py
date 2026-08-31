from boilermind.orchestration.problem_intake import analyze_problem_intake


def test_vague_problem_returns_user_choices_without_llm():
    decision = analyze_problem_intake("综合下来用哪套办法估蒸汽量最好？")
    assert decision.status == "NEEDS_CLARIFICATION"
    assert decision.missing_fields == ["target_definition"]
    assert "candidate_models" not in decision.missing_fields
    assert all(item.choices for item in decision.clarification_items)


def test_complete_supported_model_comparison_continues():
    decision = analyze_problem_intake(
        "比较Ridge、BayesianRidge、HGB与Persistence对"
        "蒸汽体积流量h80预测的MAE和RMSE。"
    )
    assert decision.status == "READY_FOR_HYPOTHESIS"
    assert decision.currently_executable


def test_frozen_dataset_is_not_misread_as_sensor_freeze_fault():
    decision = analyze_problem_intake(
        "在冻结的31变量数据上比较Ridge、BayesianRidge、HGB与"
        "Persistence对蒸汽体积流量h80预测的MAE和RMSE。"
    )
    assert decision.status == "READY_FOR_HYPOTHESIS"
    assert decision.problem_type == "model_comparison"


def test_noise_question_can_form_hypotheses_before_fault_levels_are_chosen():
    decision = analyze_problem_intake(
        "高斯噪声是否会增加蒸汽体积流量h80预测误差？"
    )
    assert decision.status == "READY_FOR_HYPOTHESIS"
    assert decision.problem_type == "sensor_fault_robustness"
    assert "corruption_levels" not in decision.missing_fields


def test_complete_direction_rate_question_defers_capability_gap_until_after_generation():
    decision = analyze_problem_intake(
        "蒸汽体积流量h80预测中，降负荷变化速率对MAE的影响"
        "是否大于升负荷变化速率？"
    )
    assert decision.status == "READY_FOR_HYPOTHESIS"
    assert "direction_rate_interaction_evaluation" in decision.required_capabilities


def test_explicit_cpu_limit_is_extracted_without_blocking_hypothesis_generation():
    decision = analyze_problem_intake(
        "比较Ridge与Persistence对蒸汽体积流量h40预测的MAE，"
        "并要求在8核CPU现场运行。"
    )
    assert decision.status == "READY_FOR_HYPOTHESIS"
    assert decision.problem_type == "resource_benchmark"
    assert decision.extracted["resource_limit"] == "8核CPU"
    assert "resource_limit" not in decision.missing_fields
