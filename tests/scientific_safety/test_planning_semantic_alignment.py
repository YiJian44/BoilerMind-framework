import pytest

from boilermind.core.contracts import ExperimentPlan
from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
    ExperimentCapabilityRegistry,
)
from boilermind.models.execution_environment import ExecutionEnvironment

from boilermind.skills.planning_skill import (
    PlanningSkill,
)
from boilermind.planning.experiment_requirement_parser import (
    freeze_hypothesis_design,
    frozen_design_sha256,
)


def make_h001():
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
            "操作，对比各模型有/无滞后特征输入时在锁定"
            "测试集上的MAE与RMSE。"
        ),
        "falsification_condition": (
            "在至少一个启用模型上，引入滞后特征后其MAE与"
            "RMSE均未降低，且该结果在chronological_validation"
            "协议下稳定复现。"
        ),
    }


def make_h002():
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
    }


def make_skill():
    capability = ExperimentCapabilityRegistry(
        environment=ExecutionEnvironment(
            os="test", python_version="3.11.9",
            sklearn_available=True, torch_available=False,
            cuda_available=False, gpu_available=False,
        )
    )
    return PlanningSkill(capability_registry=capability)


def test_h001_fails_closed_with_feature_intervention_missing():
    skill = make_skill()

    result = skill.execute(
        {
            "problem_id": "RP-XXX",
            "selected_hypothesis_id": "H001",
            "qualified_hypotheses": [
                make_h001(),
            ],
        }
    )

    assert result["current_executable"] is False
    assert result["experiment_plan"] is None
    assert result["status"] == "not_executable"

    assert (
        "operation:feature_intervention"
        in result["missing_capabilities"]
    )

    report = result["planning_report"]

    assert report["hypothesis_id"] == "H001"
    assert (
        report["experiment_type"]
        == "feature_intervention"
    )
    assert report["current_executable"] is False


def test_h002_generates_executable_plan_from_registry():
    skill = make_skill()

    result = skill.execute(
        {
            "problem_id": "RP-XXX",
            "selected_hypothesis_id": "H002",
            "qualified_hypotheses": [
                make_h002(),
            ],
        }
    )

    assert result["current_executable"] is True
    assert result["status"] == "planned"

    plan = result["experiment_plan"]

    assert plan["plan_id"] == "PLAN-H002"
    assert plan["problem_id"] == "RP-XXX"
    assert plan["hypothesis_id"] == "H002"
    assert plan["hypothesis_statement"] != ""
    assert plan["hypothesis_statement"] == make_h002()["hypothesis"]
    assert plan["experiment_type"] == "reference_model_comparison"
    assert plan["candidate_models"] == [
        "bayesianridge", "hgb", "ridge",
    ]
    assert plan["reference_models"] == ["persistence"]
    assert "ModelRegistry" in plan["model_selection_rationale"]
    assert plan["primary_metric"] == "MAE"
    assert plan["secondary_metrics"] == ["RMSE"]
    assert plan["confirmation_criteria"] == [
        "all_candidates_worse_than_reference_on:MAE,RMSE",
    ]
    assert plan["falsification_criteria"] == [
        "any_candidate_better_than_reference_on:MAE,RMSE",
    ]
    validated = ExperimentPlan.model_validate(plan)
    assert validated.hypothesis_id == "H002"
    assert validated.current_executable is True


def test_planning_blocks_frozen_design_tampering():
    skill = make_skill()
    hypothesis = make_h002()
    design = freeze_hypothesis_design(hypothesis)
    hypothesis["scientific_design"] = design.model_dump(mode="json")
    hypothesis["scientific_design_sha256"] = frozen_design_sha256(design)
    hypothesis["scientific_design"]["required_operations"].append(
        "sensor_corruption_injection"
    )

    result = skill.execute({
        "problem_id": "RP-XXX",
        "selected_hypothesis_id": "H002",
        "qualified_hypotheses": [hypothesis],
    })

    assert result["current_executable"] is False
    assert any(
        issue == "hypothesis_design:sha256_mismatch"
        for issue in result["missing_capabilities"]
    )


def test_planning_does_not_reparse_prose_after_design_freeze():
    skill = make_skill()
    hypothesis = make_h002()
    design = freeze_hypothesis_design(hypothesis)
    hypothesis["scientific_design"] = design.model_dump(mode="json")
    hypothesis["scientific_design_sha256"] = frozen_design_sha256(design)
    hypothesis["hypothesis"] = "这段展示文字已改变，但不再控制实验。"

    result = skill.execute({
        "problem_id": "RP-XXX",
        "selected_hypothesis_id": "H002",
        "qualified_hypotheses": [hypothesis],
    })

    assert result["current_executable"] is True
    assert result["experiment_plan"]["experiment_type"] == (
        design.experiment_type
    )
    assert result["experiment_plan"]["confirmation_criteria"] == (
        design.confirmation_criteria
    )


def test_frozen_atomic_design_does_not_reexpand_problem_models():
    skill = make_skill()
    hypothesis = {
        "id": "H-ATOMIC",
        "hypothesis_id": "H-ATOMIC",
        "hypothesis": (
            "bayesianridge对h80步后steam_volumetric_flow的"
            "MAE低于persistence。"
        ),
        "verification_intent": (
            "执行model_comparison和reference_model_comparison。"
        ),
        "falsification_condition": (
            "bayesianridge的MAE大于或等于persistence时证伪。"
        ),
    }
    design = freeze_hypothesis_design(hypothesis)
    hypothesis["scientific_design"] = design.model_dump(mode="json")
    hypothesis["scientific_design_sha256"] = frozen_design_sha256(design)
    result = skill.execute({
        "problem_id": "RP-ATOMIC",
        "research_problem": {
            "problem_id": "RP-ATOMIC",
            "target_variable": "steam_volumetric_flow",
            "required_models": ["ridge", "bayesianridge"],
            "reference_models": ["persistence"],
            "metrics": ["MAE"],
            "required_horizon_steps": 80,
        },
        "selected_hypothesis_id": "H-ATOMIC",
        "qualified_hypotheses": [hypothesis],
    })
    assert result["current_executable"] is True
    plan = result["experiment_plan"]
    assert plan["candidate_models"] == ["bayesianridge"]
    assert plan["recommended_models"] == ["bayesianridge"]
    assert plan["executable_models"] == ["bayesianridge"]
    assert plan["reference_models"] == ["persistence"]
    assert plan["prediction_horizon_steps"] == 80

def test_direction_rate_design_fails_closed_instead_of_model_comparison():
    hypothesis = {
        "id": "H-RATE",
        "hypothesis_id": "H-RATE",
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
    result = make_skill().execute({
        "problem_id": "RP-RATE",
        "selected_hypothesis_id": "H-RATE",
        "qualified_hypotheses": [hypothesis],
    })
    assert result["current_executable"] is False
    assert result["experiment_plan"] is None
    assert result["planning_report"]["experiment_type"] == (
        "direction_rate_interaction_evaluation"
    )
    assert "operation:direction_rate_interaction_evaluation" in (
        result["missing_capabilities"]
    )


def test_deep_peak_ramp_state_volume_question_builds_regime_plan():
    environment = ExecutionEnvironment(
        os="test",
        python_version="3.11.9",
        sklearn_available=True,
        torch_available=False,
        cuda_available=False,
        gpu_available=False,
    )
    skill = PlanningSkill(
        capability_registry=DirectVolume31VCapabilityRegistry(
            prediction_horizon_steps=40,
            environment=environment,
        )
    )
    hypothesis = {
        "id": "H-REGIME-VOLUME",
        "hypothesis_id": "H-REGIME-VOLUME",
        "hypothesis": (
            "深度调峰升负荷工况下，不同负荷变化状态会导致未来10分钟"
            "蒸汽体积流量预测MAE出现差异。"
        ),
        "verification_intent": "按ramp_up、ramp_down和steady分层比较MAE。",
        "falsification_condition": "ramp_up的MAE不高于ramp_down。",
        "verification_mapping": {
            "executable_now": True,
            "verification_scope": "problem_observable_premise_only",
        },
    }
    result = skill.execute({
        "problem_id": "RP-REGIME-VOLUME",
        "research_problem": {
            "problem_id": "RP-REGIME-VOLUME",
            "target_variable": "steam_volumetric_flow",
            "metrics": ["MAE"],
            "required_horizon_steps": 40,
        },
        "selected_hypothesis_id": hypothesis["hypothesis_id"],
        "qualified_hypotheses": [hypothesis],
    })

    assert result["current_executable"] is True
    plan = result["experiment_plan"]
    assert plan["experiment_type"] == "regime_stratified_evaluation"
    assert plan["required_operations"] == ["regime_stratified_evaluation"]
    assert plan["target"] == "steam_volumetric_flow"
    assert plan["prediction_horizon_steps"] == 40
    assert plan["candidate_models"]


def test_noise_injection_design_fails_closed_instead_of_clean_model_ranking():
    hypothesis = {
        "id": "H-NOISE",
        "hypothesis_id": "H-NOISE",
        "hypothesis": (
            "随着传感器噪声水平增加，LSTM相对Ridge的MAE优势减小。"
        ),
        "verification_intent": "注入不同强度的高斯噪声与毛刺。",
        "falsification_condition": "两类模型的误差退化斜率没有差异。",
        "variables": ["sensor_noise_level", "MAE"],
    }
    result = make_skill().execute({
        "problem_id": "RP-NOISE",
        "selected_hypothesis_id": "H-NOISE",
        "qualified_hypotheses": [hypothesis],
    })
    assert result["current_executable"] is False
    assert result["experiment_plan"] is None
    assert result["planning_report"]["experiment_type"] == (
        "sensor_corruption_robustness_evaluation"
    )
    assert "operation:gaussian_noise_injection" in result["missing_capabilities"]
    assert "operation:spike_injection" in result["missing_capabilities"]


def test_corrupted_critical_hypothesis_text_fails_closed():
    hypothesis = make_h002()
    hypothesis["hypothesis"] = "��� Ridge �� Persistence ����锛岀粨璁�"
    result = make_skill().execute({
        "problem_id": "RP-ENCODING",
        "selected_hypothesis_id": "H002",
        "qualified_hypotheses": [hypothesis],
    })
    assert result["current_executable"] is False
    assert (
        "text_encoding:critical_hypothesis_field_corrupted"
        in result["missing_capabilities"]
    )


def test_planning_requires_selected_hypothesis_id():
    skill = make_skill()

    with pytest.raises(
        ValueError,
        match="selected_hypothesis_id_required",
    ):
        skill.execute(
            {
                "qualified_hypotheses": [
                    make_h002(),
                ],
            }
        )


def test_planning_fails_closed_when_hypothesis_not_found():
    skill = make_skill()

    with pytest.raises(
        ValueError,
        match="selected_hypothesis_not_found",
    ):
        skill.execute(
            {
                "selected_hypothesis_id": "H002",
                "qualified_hypotheses": [
                    make_h001(),
                ],
            }
        )


def test_planning_rejects_model_not_in_executable_pool():
    """
    Hypothesis requires transformer, which exists in the
    ModelRegistry catalog but is NOT currently executable.
    """

    hypothesis = make_h002()
    hypothesis["hypothesis"] = (
        "在真实锅炉数据上，transformer模型对10分钟后"
        "主蒸汽流量的预测MAE低于persistence模型。"
    )

    skill = make_skill()

    result = skill.execute(
        {
            "problem_id": "RP-XXX",
            "selected_hypothesis_id": "H002",
            "qualified_hypotheses": [
                hypothesis,
            ],
        }
    )

    assert result["current_executable"] is False
    assert "model:transformer" in (
        result["missing_capabilities"]
    )


def test_planning_selects_from_executable_pool_when_models_unspecified():
    """
    Hypothesis only names persistence as reference; the
    planner must autonomously pick candidates from the
    executable pool (registry-derived, not hard-coded).
    """

    hypothesis = {
        "id": "H010",
        "hypothesis_id": "H010",
        "hypothesis": (
            "当前可执行模型对10分钟后主蒸汽流量的预测"
            "MAE与RMSE高于persistence模型。"
        ),
        "verification_intent": (
            "执行reference_model_comparison操作，在锁定"
            "测试集上计算当前可执行模型与persistence模型"
            "的MAE与RMSE。"
        ),
        "falsification_condition": (
            "至少一个当前可执行模型在MAE与RMSE上同时"
            "优于persistence模型。"
        ),
    }

    skill = make_skill()

    result = skill.execute(
        {
            "problem_id": "RP-XXX",
            "selected_hypothesis_id": "H010",
            "qualified_hypotheses": [
                hypothesis,
            ],
        }
    )

    assert result["current_executable"] is True

    plan = result["experiment_plan"]

    assert plan["candidate_models"] == [
        "bayesianridge",
        "knn",
        "pls",
    ]

    assert plan["reference_models"] == [
        "persistence",
    ]

    assert len(plan["candidate_models"]) == 3

    # High-cost models (svr / mlp / rf under the large-sample
    # contract) must NOT be picked autonomously.
    assert "svr" not in plan["candidate_models"]
    assert "mlp" not in plan["candidate_models"]
    assert "rf" not in plan["candidate_models"]

    assert "ModelRegistry" in (
        plan["model_selection_rationale"]
    )
