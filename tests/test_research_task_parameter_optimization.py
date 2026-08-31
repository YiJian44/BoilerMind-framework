from __future__ import annotations

from datetime import datetime, timezone

from boilermind.experiment.capability_registry import DirectVolume31VCapabilityRegistry
from boilermind.core.contracts import (
    ExperimentAudit,
    ExperimentResult,
    ModelExperimentRecord,
    ResearchProblemSpec,
)
from boilermind.core.enums import ExperimentStatus
from boilermind.hypothesis.hypothesis_compiler import compile_hypotheses
from boilermind.models.execution_environment import ExecutionEnvironment
from boilermind.orchestration.research_task import resolve_research_task
from boilermind.planning.parameter_optimization import (
    collect_parameter_candidate_results,
    compare_parameter_results,
    expand_parameter_plans,
)
from boilermind.skills.contract_skill import ExperimentContractSkill
from boilermind.skills.planning_skill import PlanningSkill
from boilermind.skills.ranking_skill import RankingSkill


def _capability():
    return DirectVolume31VCapabilityRegistry(environment=ExecutionEnvironment(
        os="test", python_version="3.11", sklearn_available=True,
        torch_available=False, cuda_available=False, gpu_available=False,
    ))


def _problem():
    return {
        "problem_id": "P-OPT",
        "original_question": "选择哪个时间窗口预测未来目标更准确？",
        "target_variable": "steam_volumetric_flow",
        "operating_condition": "declared operating range",
        "objective": "improve_prediction_accuracy",
        "metrics": ["MAE", "RMSE", "R2"],
        "required_horizon_steps": 40,
        "required_operations": [],
        "reference_models": [],
        "required_models": [],
        "protocol_constraints": [],
        "research_task_type": "parameter_optimization",
        "optimization_variable": "window_steps",
        "candidate_values": [10, 20, 40, 80],
    }


def _hypothesis():
    return {
        "id": "H001", "hypothesis_id": "H001",
        "title": "Window sensitivity",
        "hypothesis": "History window length changes prediction error.",
        "verification_intent": "Compare window candidates by MAE, RMSE and R2.",
        "falsification_condition": "No candidate differs on validation metrics.",
        "variables": ["steam_volumetric_flow"],
    }


def test_window_optimization_task_is_recognized():
    intent = resolve_research_task(
        "在任意工况下，选择哪个时间窗口预测未来目标更准确？"
    )
    assert intent.research_task_type == "parameter_optimization"
    assert intent.optimization_variable == "window_steps"
    assert intent.candidate_values == [10, 20, 40, 80]


def test_compiler_preserves_window_intent_and_planner_builds_four_contracts():
    capability = _capability()
    compiled, records = compile_hypotheses(
        [_hypothesis()], _problem(), capability.to_scientific_context()
    )
    selected = compiled[0]
    assert selected["experiment_intent"] == {
        "task_type": "parameter_optimization",
        "variable": "window_steps",
        "candidates": [10, 20, 40, 80],
    }
    assert records[0].supported_operations == ["model_comparison"]

    ranking = RankingSkill().execute({
        "qualified_hypotheses": compiled,
        "research_problem": _problem(),
        "scientific_context": capability.to_scientific_context(),
    })
    assert ranking["selected_hypothesis_id"] == selected["hypothesis_id"]
    assert ranking["ranking"][0]["current_executable"] is True
    assert ranking["ranking"][0]["execution_semantics_source"] == (
        "compiled_execution_intent"
    )

    planning = PlanningSkill(capability_registry=capability).execute({
        "problem_id": "P-OPT",
        "research_problem": _problem(),
        "selected_hypothesis_id": selected["hypothesis_id"],
        "qualified_hypotheses": [selected],
    })
    assert planning["current_executable"] is True
    plans = planning["experiment_plans"]
    assert [item["window_steps"] for item in plans] == [10, 20, 40, 80]
    assert [item["plan_id"] for item in plans] == [
        "PLAN-window-10", "PLAN-window-20", "PLAN-window-40", "PLAN-window-80"
    ]
    frozen = {
        key: plans[0][key]
        for key in (
            "dataset_path", "prediction_horizon_steps", "metrics",
            "required_variables", "random_seed", "candidate_models",
        )
    }
    assert all(all(plan[key] == value for key, value in frozen.items()) for plan in plans)

    contracts = [
        ExperimentContractSkill(capability_registry=capability).execute(
            {"experiment_plan": plan}
        )["experiment_contract"]
        for plan in plans
    ]
    assert [item["window_steps"] for item in contracts] == [10, 20, 40, 80]


def test_parameter_compiler_fails_closed_without_registered_operation():
    compiled, records = compile_hypotheses(
        [_hypothesis()], _problem(), {"supported_experiment_operations": []}
    )
    assert compiled[0]["compilation_status"] == "UNSUPPORTED"
    assert records[0].current_executable is False


def test_tampered_compiled_intent_fails_closed():
    capability = _capability()
    compiled, _records = compile_hypotheses(
        [_hypothesis()], _problem(), capability.to_scientific_context()
    )
    compiled[0]["scientific_design"]["required_operations"] = []
    ranking = RankingSkill().execute({
        "qualified_hypotheses": compiled,
        "research_problem": _problem(),
        "scientific_context": capability.to_scientific_context(),
    })
    assert ranking["selected_hypothesis_id"] is None
    assert any(
        "compiled_execution_intent_sha256_mismatch" in item
        for item in ranking["selection_blockers"]
    )


def test_parameter_variant_keeps_raw_claim_but_removes_unsupported_numbers():
    capability = _capability()
    hypothesis = _hypothesis()
    hypothesis["hypothesis"] = "A window improves error by 5 percent."
    hypothesis["hypothesis_statement"] = hypothesis["hypothesis"]
    compiled, _records = compile_hypotheses(
        [hypothesis], _problem(), capability.to_scientific_context()
    )
    variant = compiled[0]
    assert "5" not in variant["hypothesis"]
    assert "5" in variant["original_claim"]
    assert variant["experiment_intent"]["variable"] == "window_steps"


def test_parameter_result_uses_validation_metric_and_selects_best():
    result = compare_parameter_results([
        {"candidate": 10, "validation_metrics": {"MAE": 4.0}},
        {"candidate": 20, "validation_metrics": {"MAE": 3.0}},
        {"candidate": 40, "validation_metrics": {"MAE": 1.5}},
        {"candidate": 80, "validation_metrics": {"MAE": 2.0}},
    ], variable="window_steps", candidates=[10, 20, 40, 80])
    assert result.best_candidate == 40
    assert result.selection_metric == "MAE"
    assert result.status == "PASS"


def test_parameter_result_normalizes_planning_validation_metric_identifier():
    result = compare_parameter_results([
        {"candidate": 10, "validation_metrics": {"mae_t_h": 4.0}},
        {"candidate": 20, "validation_metrics": {"mae_t_h": 3.0}},
        {"candidate": 40, "validation_metrics": {"mae_t_h": 1.5}},
        {"candidate": 80, "validation_metrics": {"mae_t_h": 2.0}},
    ], variable="window_steps", candidates=[10, 20, 40, 80],
       selection_metric="validation_mae_t_h")
    assert result.best_candidate == 40
    assert result.selection_metric == "MAE"
    assert result.status == "PASS"


def test_locked_test_metrics_never_change_parameter_selection():
    result = compare_parameter_results([
        {"candidate": 10, "validation_metrics": {"MAE": 4.0},
         "locked_test_metrics": {"MAE": 0.01}},
        {"candidate": 40, "validation_metrics": {"MAE": 1.5},
         "locked_test_metrics": {"MAE": 100.0}},
    ], variable="window_steps", candidates=[10, 40])
    assert result.best_candidate == 40


def test_parameter_selection_fails_closed_without_validation_metrics():
    result = compare_parameter_results([
        {"candidate": 10, "validation_metrics": {},
         "locked_test_metrics": {"MAE": 0.01}},
        {"candidate": 40, "locked_test_metrics": {"MAE": 0.02}},
    ], variable="window_steps", candidates=[10, 40])
    assert result.best_candidate is None
    assert result.confidence == 0.0
    assert result.status == "INSUFFICIENT_VALIDATION_EVIDENCE"
    assert result.reason == "no_validation_MAE_available"


def test_parameter_result_maximizes_validation_r2():
    result = compare_parameter_results([
        {"candidate": 10, "validation_metrics": {"R2": 0.70}},
        {"candidate": 20, "validation_metrics": {"R2": 0.80}},
        {"candidate": 40, "validation_metrics": {"R2": 0.90}},
        {"candidate": 80, "validation_metrics": {"R2": 0.85}},
    ], variable="window_steps", candidates=[10, 20, 40, 80],
       selection_metric="validation_r2")
    assert result.best_candidate == 40
    assert result.selection_metric == "R2"


def test_four_round_experiment_results_produce_complete_best_candidate():
    candidates = [10, 20, 40, 80]
    validation_mae = [4.0, 3.0, 1.5, 2.0]
    locked_mae = [0.01, 0.02, 100.0, 0.03]
    plans = [{"window_steps": candidate} for candidate in candidates]
    outcomes = []
    now = datetime(2026, 8, 24, tzinfo=timezone.utc)
    for candidate, validation, locked in zip(
        candidates, validation_mae, locked_mae, strict=True
    ):
        experiment_id = f"EXP-window-{candidate}"
        result = ExperimentResult(
            experiment_id=experiment_id,
            problem_id="P-OPT",
            hypothesis_id="H001",
            plan_id=f"PLAN-window-{candidate}",
            status=ExperimentStatus.COMPLETED,
            metrics={"MAE": locked},
            model_records={
                "ridge": ModelExperimentRecord(
                    model_name="ridge",
                    fit_success=True,
                    fit_converged=True,
                    validation_metrics={"mae_t_h": validation},
                    locked_test_metrics={"MAE": locked},
                )
            },
            started_at=now,
            completed_at=now,
        )
        audit = ExperimentAudit(
            experiment_id=experiment_id,
            execution_valid=True,
            dataset_frozen=True,
            leakage_check_passed=True,
            baseline_valid=True,
            metric_check_passed=True,
        )
        outcomes.append({"experiment_result": result, "audit": audit})

    candidate_results = collect_parameter_candidate_results(
        plans,
        outcomes,
        variable="window_steps",
        selection_metric="validation_mae_t_h",
    )
    optimization = compare_parameter_results(
        candidate_results,
        variable="window_steps",
        candidates=candidates,
        selection_metric="validation_mae_t_h",
    )

    assert [item["candidate"] for item in candidate_results] == candidates
    assert all(item["selected_model"] == "ridge" for item in candidate_results)
    assert all("MAE" in item["validation_metrics"] for item in candidate_results)
    assert candidate_results[2]["experiment_id"] == "EXP-window-40"
    assert candidate_results[2]["locked_test_metrics"]["MAE"] == 100.0
    assert optimization.best_candidate == 40
    assert optimization.selection_metric == "MAE"
    assert optimization.status == "PASS"


def test_invalid_candidate_is_not_selected():
    result = compare_parameter_results([
        {"candidate": 10, "experiment_valid": False,
         "validation_metrics": {"MAE": 0.1}},
        {"candidate": 20, "experiment_valid": True,
         "validation_metrics": {"MAE": 2.0}},
    ], variable="window_steps", candidates=[10, 20])
    assert result.best_candidate == 20


def test_existing_hypothesis_validation_remains_default():
    intent = resolve_research_task("验证一个预声明科研假设")
    assert intent.research_task_type == "hypothesis_validation"


def test_research_problem_old_payload_remains_compatible():
    problem = ResearchProblemSpec.model_validate({
        "problem_id": "P-OLD",
        "original_question": "Validate a claim",
        "research_object": "generic process",
        "target_variable": "generic_target",
        "operating_condition": "declared range",
        "research_goal": "validate",
    })
    assert problem.research_task_type == "hypothesis_validation"
    assert problem.optimization_variable is None
    assert problem.candidate_values == []
