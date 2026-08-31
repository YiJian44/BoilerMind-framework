from boilermind.orchestration.target_normalizer import normalize_target_variable


def test_volume_target_is_extracted_from_task_description() -> None:
    result = normalize_target_variable(
        "深度调峰升负荷工况下，未来10分钟蒸汽体积流量预测",
        "深度调峰升负荷工况下，未来10分钟蒸汽体积流量预测",
    )

    assert result.normalized_target_variable == "steam_volumetric_flow"


def test_mass_flow_target_is_normalized() -> None:
    result = normalize_target_variable(
        "主蒸汽质量流量预测",
        "主蒸汽质量流量预测",
    )

    assert result.normalized_target_variable == "main_steam_mass_flow"


def test_unknown_or_corrupted_target_fails_closed() -> None:
    result = normalize_target_variable(
        "δ��10���������������",
        "研究未知对象的变化规律",
    )

    assert result.normalized_target_variable == "unspecified"
    assert result.normalization_reason == "target_variable_resolution_failed"


def test_corrupted_raw_target_can_use_unique_original_question_alias() -> None:
    result = normalize_target_variable(
        "δ��10���������������",
        "不同负荷状态对未来10分钟蒸汽体积流量预测的影响",
    )

    assert result.normalized_target_variable == "steam_volumetric_flow"
    assert result.normalization_reason == "unique_alias_in_original_question"
