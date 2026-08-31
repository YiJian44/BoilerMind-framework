import copy

import pytest

from boilermind.skills.contract_skill import (
    ExperimentContractSkill,
)

from boilermind.skills.planning_skill import (
    PlanningSkill,
)


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


def make_h001():
    return {
        "id": "H001",
        "hypothesis_id": "H001",
        "hypothesis": (
            "为ridge、bayesianridge和hgb模型引入时间序列"
            "滞后特征后，10分钟后主蒸汽流量预测MAE与RMSE"
            "将低于未使用滞后特征的基线模型。"
        ),
        "verification_intent": (
            "执行model_comparison，对比有/无滞后特征输入"
            "时在锁定测试集上的MAE与RMSE。"
        ),
        "falsification_condition": (
            "引入滞后特征后其MAE与RMSE均未降低。"
        ),
    }


def plan_h002(
    problem_id="RP-XXX",
):
    skill = PlanningSkill()

    result = skill.execute(
        {
            "problem_id": problem_id,
            "selected_hypothesis_id": "H002",
            "qualified_hypotheses": [
                make_h002(),
            ],
        }
    )

    assert result["current_executable"] is True

    return result["experiment_plan"]


def compile_contract(plan):
    skill = ExperimentContractSkill()

    return skill.execute(
        {
            "experiment_plan": plan,
        }
    )


def test_full_id_chain_plan_to_contract():
    plan = plan_h002("RP-XXX")

    result = compile_contract(plan)

    assert result["contract_compiled"] is True
    assert result["status"] == "contract_ready"

    contract = result["experiment_contract"]

    assert contract["problem_id"] == "RP-XXX"
    assert contract["hypothesis_id"] == "H002"
    assert contract["plan_id"] == "PLAN-H002"

    assert contract["experiment_id"]
    assert contract["experiment_id"].startswith(
        "EXP-"
    )

    assert (
        plan["problem_id"]
        == contract["problem_id"]
    )
    assert (
        plan["hypothesis_id"]
        == contract["hypothesis_id"]
    )
    assert (
        plan["plan_id"]
        == contract["plan_id"]
    )


def test_contract_preserves_h002_semantics():
    plan = plan_h002("RP-XXX")

    result = compile_contract(plan)

    contract = result["experiment_contract"]

    assert (
        contract["experiment_type"]
        == "reference_model_comparison"
    )

    assert contract["candidate_models"] == [
        "bayesianridge",
        "hgb",
        "ridge",
    ]

    assert contract["reference_models"] == [
        "persistence",
    ]

    assert (
        contract["prediction_horizon_steps"]
        == 40
    )

    assert (
        contract["sampling_interval_seconds"]
        == 15
    )

    assert contract["primary_metric"] == "MAE"
    assert contract["secondary_metrics"] == ["RMSE"]

    assert contract["confirmation_criteria"] == [
        "all_candidates_worse_than_reference_on:"
        "MAE,RMSE",
    ]

    assert contract["falsification_criteria"] == [
        "any_candidate_better_than_reference_on:"
        "MAE,RMSE",
    ]

    assert (
        contract["locked_test_used_for_selection"]
        is False
    )


def test_problem_id_from_research_problem():
    skill = PlanningSkill()

    result = skill.execute(
        {
            "research_problem": {
                "problem_id": "RP-FROM-RESEARCH",
            },
            "selected_hypothesis_id": "H002",
            "qualified_hypotheses": [
                make_h002(),
            ],
        }
    )

    plan = result["experiment_plan"]

    assert (
        plan["problem_id"]
        == "RP-FROM-RESEARCH"
    )


def test_planning_fails_closed_without_problem_id():
    skill = PlanningSkill()

    with pytest.raises(
        ValueError,
        match="problem_id_required",
    ):
        skill.execute(
            {
                "selected_hypothesis_id": "H002",
                "qualified_hypotheses": [
                    make_h002(),
                ],
            }
        )


def test_h001_plan_cannot_produce_contract():
    """
    H001 is not executable; planning returns no
    experiment_plan, and even a direct contract_skill
    call with a tampered non-executable plan fails.
    """

    planning = PlanningSkill()

    planning_result = planning.execute(
        {
            "problem_id": "RP-XXX",
            "selected_hypothesis_id": "H001",
            "qualified_hypotheses": [
                make_h001(),
            ],
        }
    )

    assert planning_result["current_executable"] is False
    assert planning_result["experiment_plan"] is None
    assert (
        "operation:feature_intervention"
        in planning_result["missing_capabilities"]
    )

    with pytest.raises(
        ValueError,
        match="experiment_plan_required",
    ):
        compile_contract(
            planning_result["experiment_plan"]
        )


def test_tampered_non_executable_plan_fails_at_gate():
    plan = plan_h002("RP-XXX")

    tampered = copy.deepcopy(plan)
    tampered["current_executable"] = False
    tampered["missing_capabilities"] = [
        "operation:feature_intervention",
    ]

    result = compile_contract(tampered)

    assert result["contract_compiled"] is False
    assert result["experiment_contract"] is None
    assert "plan_not_currently_executable" in (
        result["issues"]
    )
    assert any(
        issue.startswith(
            "plan_missing_capabilities"
        )
        for issue in result["issues"]
    )


def test_gate_rejects_empty_candidate_models():
    plan = plan_h002("RP-XXX")

    tampered = copy.deepcopy(plan)
    tampered["candidate_models"] = []
    tampered["model_candidates"] = []

    result = compile_contract(tampered)

    assert result["contract_compiled"] is False
    assert "candidate_models_required" in (
        result["issues"]
    )


def test_gate_rejects_locked_test_selection():
    plan = plan_h002("RP-XXX")

    tampered = copy.deepcopy(plan)
    tampered["locked_test_used_for_selection"] = True

    result = compile_contract(tampered)

    assert result["contract_compiled"] is False
    assert (
        "locked_test_used_for_selection_forbidden"
        in result["issues"]
    )


def test_contract_experiment_id_is_new():
    contract_a = compile_contract(
        plan_h002("RP-XXX")
    )["experiment_contract"]

    contract_b = compile_contract(
        plan_h002("RP-XXX")
    )["experiment_contract"]

    assert (
        contract_a["experiment_id"]
        != contract_b["experiment_id"]
    )
