from boilermind.models import build_default_registry
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.skills.contract_skill import ExperimentContractSkill
from boilermind.skills.model_selection_skill import ModelSelectionSkill
from boilermind.skills.planning_skill import PlanningSkill


HYPOTHESIS_TEXT = (
    "增加历史实验窗口数，是否可以改善调峰工况下"
    "软测量模型预测精度"
)


def _hypothesis():
    return {
        "hypothesis_id": "H-WINDOW",
        "hypothesis": HYPOTHESIS_TEXT,
        "verification_intent": (
            "执行model_comparison与chronological_validation，"
            "比较不同历史窗口下候选模型在锁定测试集的MAE与RMSE"
        ),
        "falsification_condition": (
            "增加历史窗口后MAE与RMSE均未改善"
        ),
        "applicability_conditions": ["深度调峰工况"],
    }


def test_model_selection_returns_only_registered_models():
    registry = build_default_registry()
    result = ModelSelectionSkill(model_registry=registry).execute({
        "hypothesis": _hypothesis(),
        "target_variable": "main_steam_mass_flow",
        "operating_condition": "深度调峰工况",
    })

    assert result["candidate_models"]
    assert result["baseline_models"]
    assert set(result["candidate_models"] + result["baseline_models"]).issubset(
        set(registry.names())
    )
    assert "ModelRegistry" in result["selection_reason"]


def test_registry_exposes_scientific_capability_tags():
    registry = build_default_registry()
    assert "temporal_dependency" in registry.get("dlinear").capability_tags
    assert "physics_constraint" in registry.get("psfa_v20").capability_tags
    assert "feature_selection" in registry.get("elasticnet").capability_tags
    assert "optimization_surrogate" in registry.get("ridge").capability_tags


def test_steam_prediction_returns_prediction_models():
    registry = build_default_registry()
    result = ModelSelectionSkill(model_registry=registry).select_models(
        hypothesis_statement=HYPOTHESIS_TEXT,
        mechanism_chain="历史窗口长度影响动态信息覆盖",
        verification_intent="比较不同时序窗口",
        task_type="prediction",
        target_variable="main_steam_mass_flow",
        operating_condition="深度调峰",
        objective="预测未来蒸汽量",
    )
    assert result["task_type"] == "prediction"
    assert result["recommended_models"] == [
        "dlinear", "gru", "itransformer",
    ]
    assert "itransformer" in result["recommended_models"]
    assert "itransformer" not in result["executable_models"]
    assert set(result["executable_models"]).issubset(
        set(ExperimentCapabilityRegistry().available_models())
    )
    assert result["candidate_models"]
    for name in result["candidate_models"]:
        assert set(registry.get(name).task_list) & {
            "prediction", "soft_sensor_prediction",
            "mass_flow_forecast", "steam_volume_forecast",
        }
    assert "sequence_model" in result["required_capability_tags"]
    assert all(
        "sequence_model" in registry.get(name).capability_tags
        for name in result["candidate_models"]
    )


def test_nox_problem_returns_registered_regression_models():
    registry = build_default_registry()
    result = ModelSelectionSkill(model_registry=registry).select_models(
        hypothesis_statement="利用过程变量预测NOx",
        mechanism_chain="燃烧状态影响NOx生成",
        verification_intent="比较回归模型的NOx预测误差",
        task_type="prediction",
        target_variable="NOx",
        operating_condition="低负荷深度调峰",
        objective="建立NOx软测量预测模型",
    )
    assert result["canonical_target"] == "NOx"
    assert result["candidate_models"] == []
    assert result["recommended_models"]
    assert "No currently executable models" in result["model_substitution_reason"]
    selected = (
        result["recommended_models"]
        + result["recommended_baseline_models"]
    )
    assert set(selected).issubset(set(registry.names()))
    assert all("NOx" in registry.get(name).supported_targets for name in selected)


def test_coal_air_optimization_returns_registered_surrogate_models():
    registry = build_default_registry()
    result = ModelSelectionSkill(model_registry=registry).select_models(
        hypothesis_statement="给煤与送风组合影响锅炉效率",
        mechanism_chain="给煤送风→燃烧状态→效率",
        verification_intent="拟合代理模型比较候选工况",
        task_type="optimization",
        target_variable="给煤量与送风量",
        operating_condition="深度调峰",
        objective="优化给煤量和送风量以提高效率",
    )
    assert result["task_type"] == "optimization"
    assert result["canonical_target"] == "optimization_objective"
    assert result["candidate_models"] == []
    assert result["recommended_models"]
    assert "No currently executable models" in result["model_substitution_reason"]
    assert "optimization surrogates" in result["selection_reason"]
    selected = (
        result["recommended_models"]
        + result["recommended_baseline_models"]
    )
    assert set(selected).issubset(set(registry.names()))
    assert all(
        "optimization_surrogate" in registry.get(name).task_list
        for name in selected
    )


def test_diagnosis_without_registered_classifier_fails_closed():
    result = ModelSelectionSkill().select_models(
        hypothesis_statement="诊断水冷壁故障",
        mechanism_chain="异常征兆→故障",
        verification_intent="执行故障分类",
        task_type="diagnosis",
        target_variable="water_wall_fault",
        operating_condition="深度调峰",
        objective="诊断水冷壁故障",
    )
    assert result["candidate_models"] == []
    assert result["baseline_models"] == []


def test_window_hypothesis_plan_and_contract_keep_registry_selection():
    result = PlanningSkill().execute({
        "problem_id": "P-WINDOW",
        "research_problem": {
            "problem_id": "P-WINDOW",
            "target_variable": "main_steam_mass_flow",
            "objective": "improve_prediction_accuracy",
            "metrics": ["MAE", "RMSE", "R2"],
        },
        "selected_hypothesis_id": "H-WINDOW",
        "qualified_hypotheses": [_hypothesis()],
    })

    assert result["current_executable"] is True
    plan = result["experiment_plan"]
    assert plan["candidate_models"]
    assert plan["reference_models"]
    assert plan["recommended_models"] == [
        "dlinear", "gru", "itransformer",
    ]
    assert plan["executable_models"] == plan["candidate_models"]
    assert plan["recommended_models"] != plan["executable_models"]
    assert "preserved" in plan["model_substitution_reason"]
    assert plan["model_candidates"] == plan["candidate_models"]
    assert plan["target"] == "main_steam_mass_flow"
    assert plan["objective"] == "improve_prediction_accuracy"
    assert plan["metrics"] == ["MAE", "RMSE", "R2"]
    assert plan["model_selection_rationale"]
    assert "temporal_dependency" in plan["model_selection_rationale"]
    assert all(
        name in plan["model_selection_rationale"]
        for name in plan["recommended_models"]
    )

    registry_names = set(build_default_registry().names())
    assert set(plan["candidate_models"] + plan["reference_models"]).issubset(
        registry_names
    )

    compiled = ExperimentContractSkill().execute({"experiment_plan": plan})
    assert compiled["contract_compiled"] is True
    contract = compiled["experiment_contract"]
    assert contract["hypothesis_id"] == "H-WINDOW"
    assert contract["plan_id"] == "PLAN-H-WINDOW"
    assert contract["dataset_id"]
    assert contract["candidate_models"] == plan["candidate_models"]
    assert contract["baseline_models"] == plan["reference_models"]
    assert contract["model_selection_rationale"] == (
        plan["model_selection_rationale"]
    )
    assert contract["recommended_models"] == plan["recommended_models"]
    assert contract["executable_models"] == plan["executable_models"]
    assert contract["model_substitution_reason"] == (
        plan["model_substitution_reason"]
    )
    assert contract["target_variable"]
    assert contract["input_variables"]
    assert contract["window_steps"] == plan["window_steps"]
    assert contract["metrics"] == plan["metrics"]
