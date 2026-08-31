import pytest

from boilermind.skills.model_selection_skill import ModelSelectionSkill
from boilermind.skills.target_resolution_skill import TargetResolutionSkill


def test_soft_sensor_target_resolves_from_declared_dataset_schema():
    result = TargetResolutionSkill().execute({
        "research_problem": {
            "original_question": (
                "增加历史实验窗口数，是否可以改善调峰工况下"
                "软测量模型预测精度"
            ),
            "research_object": "深度调峰锅炉软测量",
            "target_variable": "unspecified",
            "objective": "improve_prediction_accuracy",
            "metrics": ["MAE", "RMSE", "R2"],
        },
        "task_type": "prediction",
        "current_data_schema": {
            "target_columns": ["main_steam_mass_flow"],
            "columns": ["feature_01", "main_steam_mass_flow"],
        },
    })

    assert result["resolved"] is True
    assert result["target_variable"] == "main_steam_mass_flow"
    assert result["objective"] == "improve_prediction_accuracy"
    assert result["metrics"] == ["MAE", "RMSE", "R2"]
    assert result["target_inference_reason"]


def test_unknown_task_does_not_receive_a_fixed_target():
    result = TargetResolutionSkill().execute({
        "research_problem": {
            "original_question": "研究一个尚未定义的未知任务",
            "research_object": "unknown system",
            "target_variable": "unspecified",
            "objective": "unspecified",
            "metrics": [],
        },
        "task_type": "unknown",
        "current_data_schema": {"target_columns": ["measured_output"]},
    })

    assert result["resolved"] is False
    assert result["target_variable"] == "unspecified"
    assert result["status"] == "target_variable_resolution_failed"


def test_model_selection_rejects_unresolved_target():
    with pytest.raises(ValueError, match="target_variable_resolution_required"):
        ModelSelectionSkill().select_models(
            hypothesis_statement="unknown",
            mechanism_chain="unknown",
            verification_intent="unknown",
            task_type="prediction",
            target_variable="unspecified",
        )


def test_target_can_be_read_from_role_annotated_schema():
    result = TargetResolutionSkill().execute({
        "research_problem": {
            "original_question": "使用软测量预测提高预测精度",
            "research_object": "industrial process",
            "target_variable": "unspecified",
            "objective": "improve_prediction_accuracy",
            "metrics": ["MAE"],
        },
        "task_type": "prediction",
        "current_data_schema": {
            "columns": [
                {"name": "sensor_a", "role": "feature"},
                {"name": "process_output", "role": "target"},
            ],
        },
    })
    assert result["target_variable"] == "process_output"


def test_optimization_target_resolves_from_schema_objective():
    result = TargetResolutionSkill().execute({
        "research_problem": {
            "original_question": "优化给煤量和送风量以提高锅炉效率",
            "research_object": "boiler operation optimization",
            "target_variable": "unspecified",
            "objective": "maximize_boiler_efficiency",
            "metrics": [],
        },
        "task_type": "optimization",
        "current_data_schema": {
            "target_columns": ["optimization_objective"],
            "metrics": ["efficiency"],
        },
    })

    assert result["resolved"] is True
    assert result["target_variable"] == "optimization_objective"
    assert result["objective"] == "maximize_boiler_efficiency"
    assert result["metrics"] == ["efficiency"]
