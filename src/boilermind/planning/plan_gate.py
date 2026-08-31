from copy import deepcopy
from pathlib import Path
import re
from uuid import uuid4

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentPlan,
    ScientificHypothesis,
)

from boilermind.planning.plan_contracts import (
    ExperimentCapabilitySnapshot,
    PlanApprovalReport,
    PlanCritiqueDecision,
)


class PlanAdmissionError(ValueError):
    pass


def _plan_capability_issues(
    plan: ExperimentPlan,
    capability: ExperimentCapabilitySnapshot,
    *,
    target_variable: str,
    baseline_models: list[str],
    candidate_models: list[str],
) -> list[str]:
    """
    Shared plan-vs-capability checks used by both the legacy
    critique-based gate and the P0-4 plan gate.
    """

    issues: list[str] = []

    if plan.experiment_type in {"feature_ablation", "feature_intervention"}:
        control_model = plan.control.get("model") or plan.control.get("model_name")
        treatment_model = plan.treatment.get("model") or plan.treatment.get("model_name")
        control_features = plan.control.get("features")
        treatment_features = plan.treatment.get("features")
        if not control_model or not treatment_model:
            issues.append("intervention_group_model_required")
        elif control_model != treatment_model:
            issues.append("intervention_requires_same_model")
        if not isinstance(control_features, list) or not control_features:
            issues.append("control_features_required")
        if not isinstance(treatment_features, list) or not treatment_features:
            issues.append("treatment_features_required")
        if isinstance(control_features, list) and isinstance(treatment_features, list):
            if control_features == treatment_features:
                issues.append("control_treatment_features_must_differ")
            for feature in dict.fromkeys(control_features + treatment_features):
                if feature not in capability.available_variables:
                    issues.append(f"unavailable_intervention_feature:{feature}")
        planned = list(plan.candidate_models or plan.model_candidates)
        if control_model and planned != [control_model]:
            issues.append("intervention_candidate_model_mismatch")

    available_variables = set(
        capability.available_variables
    )

    missing_variables = [
        variable
        for variable in plan.required_variables
        if variable not in available_variables
    ]

    for variable in missing_variables:
        issues.append(
            f"missing_required_variable:{variable}"
        )

    if (
        target_variable
        not in capability.available_target_variables
    ):
        issues.append(
            f"unavailable_target_variable:"
            f"{target_variable}"
        )

    available_metrics = set(
        capability.available_metrics
    )

    for metric in plan.metrics:
        if metric not in available_metrics:
            issues.append(
                f"unsupported_metric:{metric}"
            )

    available_baselines = set(
        capability.available_baseline_models
    )

    for model in baseline_models:
        if model not in available_baselines:
            issues.append(
                f"unavailable_baseline_model:{model}"
            )

    available_candidates = set(
        capability.available_candidate_models
    )

    for model in candidate_models:
        if model not in available_candidates:
            issues.append(
                f"unavailable_candidate_model:{model}"
            )

    if not baseline_models:
        issues.append(
            "baseline_models_required"
        )

    if not candidate_models:
        issues.append(
            "candidate_models_required"
        )

    if plan.recommended_models or plan.executable_models:
        if not plan.recommended_models:
            issues.append("recommended_models_required")
        if not plan.executable_models:
            issues.append("executable_models_required")
        if list(candidate_models) != list(plan.executable_models):
            issues.append("candidate_models_must_equal_executable_models")
        if (
            plan.recommended_models != plan.executable_models
            and not plan.model_substitution_reason.strip()
        ):
            issues.append("model_substitution_reason_required")

    if not capability.data_frozen:
        issues.append(
            "dataset_not_frozen"
        )

    if not capability.leakage_policy_verified:
        issues.append(
            "leakage_policy_not_verified"
        )

    if not plan.confirmation_criteria:
        issues.append(
            "confirmation_criteria_required"
        )

    if not plan.falsification_criteria:
        issues.append(
            "falsification_criteria_required"
        )

    return issues


def _build_contract(
    plan: ExperimentPlan,
    capability: ExperimentCapabilitySnapshot,
    *,
    target_variable: str,
    baseline_models: list[str],
    candidate_models: list[str],
    experiment_id: str,
) -> ExperimentContract:

    return ExperimentContract(
        experiment_id=experiment_id,
        problem_id=plan.problem_id,
        hypothesis_id=plan.hypothesis_id,
        plan_id=plan.plan_id,
        hypothesis_binding=deepcopy(plan.hypothesis_binding),
        experiment_type=plan.experiment_type,
        control=deepcopy(plan.control),
        treatment=deepcopy(plan.treatment),
        primary_metric=plan.primary_metric,
        secondary_metrics=list(
            plan.secondary_metrics
        ),
        reference_models=list(
            plan.reference_models
        ),
        model_selection_rationale=(
            plan.model_selection_rationale
        ),
        recommended_models=list(plan.recommended_models),
        executable_models=list(plan.executable_models),
        model_substitution_reason=plan.model_substitution_reason,
        target_inference_reason=plan.target_inference_reason,
        prediction_horizon_steps=(
            plan.prediction_horizon_steps
        ),
        sampling_interval_seconds=(
            plan.sampling_interval_seconds
        ),
        window_steps=plan.window_steps,
        locked_test_used_for_selection=(
            plan.locked_test_used_for_selection
        ),
        required_operations=list(plan.required_operations),
        constraints=list(plan.hard_constraints),
        execution_requirements=dict(plan.execution_requirements),
        allow_partial_failure=plan.allow_partial_failure,
        max_runtime_per_model=plan.max_runtime_per_model,
        max_epochs=plan.max_epochs,
        allowed_devices=list(plan.allowed_devices),
        reuse_checkpoint_models=list(plan.reuse_checkpoint_models),
        dataset_id=capability.dataset_id,
        dataset_hash=capability.dataset_hash,
        input_variables=list(
            plan.required_variables
        ),
        target_variable=target_variable,
        train_split=capability.train_split,
        validation_split=(
            capability.validation_split
        ),
        test_split=capability.test_split,
        baseline_models=list(baseline_models),
        candidate_models=list(candidate_models),
        metrics=list(plan.metrics),
        confirmation_criteria=list(
            plan.confirmation_criteria
        ),
        falsification_criteria=list(
            plan.falsification_criteria
        ),
        random_seed=plan.random_seed,
    )


def compile_plan_to_contract(
    plan: ExperimentPlan,
    capability: ExperimentCapabilitySnapshot,
    *,
    target_variable: str,
    baseline_models: list[str],
    candidate_models: list[str],
) -> tuple[
    ExperimentContract | None,
    PlanApprovalReport,
]:
    """
    P0-4 gate: compile an already-planned ExperimentPlan into
    a real ExperimentContract, fail closed on any ID loss or
    capability mismatch.

    A non-executable plan (current_executable=False or
    missing_capabilities non-empty) can NEVER become a
    contract here, even if called directly.
    """

    issues: list[str] = []

    if list(candidate_models) != list(plan.candidate_models or plan.model_candidates):
        issues.append("candidate_models_must_come_from_plan")
    if list(baseline_models) != list(plan.reference_models or ([plan.reference_model] if plan.reference_model else [])):
        issues.append("reference_models_must_come_from_plan")

    # ------------------------------------------------
    # 1. ID chain must be complete
    # ------------------------------------------------

    if not plan.problem_id:
        issues.append(
            "problem_id_required"
        )

    if not plan.hypothesis_id:
        issues.append(
            "hypothesis_id_required"
        )

    if not plan.plan_id:
        issues.append(
            "plan_id_required"
        )

    binding = plan.hypothesis_binding
    if binding:
        if binding.get("hypothesis_id") != plan.hypothesis_id:
            issues.append("hypothesis_binding_id_mismatch")
        binding_hash = str(binding.get("immutable_hypothesis_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", binding_hash):
            issues.append("immutable_hypothesis_sha256_required")
        for field in (
            "expected_observation", "experiment_measurement",
            "confirmation_mapping", "falsification_mapping",
        ):
            if not binding.get(field):
                issues.append(f"hypothesis_binding_{field}_required")

    # ------------------------------------------------
    # 2. Executable gate (cannot be bypassed)
    # ------------------------------------------------

    if not plan.current_executable:
        issues.append(
            "plan_not_currently_executable"
        )

    if plan.missing_capabilities:
        issues.append(
            "plan_missing_capabilities:"
            + ",".join(plan.missing_capabilities)
        )

    # ------------------------------------------------
    # 3. Experiment semantics must be complete
    # ------------------------------------------------

    if not plan.candidate_models:
        issues.append(
            "candidate_models_required"
        )

    if not plan.reference_models:
        issues.append(
            "reference_models_required"
        )

    if (
        plan.prediction_horizon_steps is not None
        and plan.prediction_horizon_steps < 0
    ):
        issues.append(
            "invalid_prediction_horizon_steps"
        )

    if (
        capability.prediction_horizon_steps is not None
        and plan.prediction_horizon_steps is not None
        and plan.prediction_horizon_steps
        != capability.prediction_horizon_steps
    ):
        issues.append(
            "prediction_horizon_mismatch_"
            "with_capability"
        )

    if (
        capability.sampling_interval_seconds is not None
        and plan.sampling_interval_seconds is not None
        and plan.sampling_interval_seconds
        != capability.sampling_interval_seconds
    ):
        issues.append(
            "sampling_interval_mismatch_"
            "with_capability"
        )

    if plan.locked_test_used_for_selection:
        issues.append(
            "locked_test_used_for_selection_forbidden"
        )

    if (
        capability.dataset_path
        and not Path(
            capability.dataset_path
        ).is_file()
    ):
        issues.append(
            "dataset_path_not_found"
        )

    # ------------------------------------------------
    # 4. Shared plan-vs-capability checks
    # ------------------------------------------------

    issues.extend(
        _plan_capability_issues(
            plan,
            capability,
            target_variable=target_variable,
            baseline_models=baseline_models,
            candidate_models=candidate_models,
        )
    )

    issues = list(
        dict.fromkeys(issues)
    )

    if issues:
        return (
            None,
            PlanApprovalReport(
                plan_id=plan.plan_id,
                hypothesis_id=plan.hypothesis_id,
                passed=False,
                issues=issues,
                experiment_id=None,
            ),
        )

    experiment_id = (
        f"EXP-{uuid4().hex[:12]}"
    )

    contract = _build_contract(
        plan,
        capability,
        target_variable=target_variable,
        baseline_models=baseline_models,
        candidate_models=candidate_models,
        experiment_id=experiment_id,
    )

    return (
        contract,
        PlanApprovalReport(
            plan_id=plan.plan_id,
            hypothesis_id=plan.hypothesis_id,
            passed=True,
            issues=[],
            experiment_id=experiment_id,
        ),
    )


def approve_and_compile_plan(
    hypothesis: ScientificHypothesis,
    plan: ExperimentPlan,
    capability: ExperimentCapabilitySnapshot,
    critique: PlanCritiqueDecision,
    *,
    target_variable: str,
    baseline_models: list[str],
    candidate_models: list[str],
) -> tuple[
    ExperimentContract | None,
    PlanApprovalReport,
]:
    """
    Legacy critique-based gate. Kept unchanged so existing
    tests and callers keep working.
    """

    issues: list[str] = []

    # -------------------------------------------------
    # 1. Identity consistency
    # -------------------------------------------------

    if plan.hypothesis_id != hypothesis.hypothesis_id:
        issues.append(
            "plan_hypothesis_id_mismatch"
        )

    if critique.plan_id != plan.plan_id:
        issues.append(
            "critique_plan_id_mismatch"
        )

    if critique.hypothesis_id != hypothesis.hypothesis_id:
        issues.append(
            "critique_hypothesis_id_mismatch"
        )

    # -------------------------------------------------
    # 2. Scientific alignment checks
    # -------------------------------------------------

    scientific_checks = {
        "hypothesis_experiment_misalignment": (
            critique.hypothesis_experiment_alignment
        ),
        "invalid_intervention": (
            critique.intervention_valid
        ),
        "invalid_baseline": (
            critique.baseline_valid
        ),
        "metric_misalignment": (
            critique.metric_alignment
        ),
        "invalid_confirmation_falsification": (
            critique.confirmation_falsification_valid
        ),
        "plan_not_executable": (
            critique.executable
        ),
    }

    for issue_name, passed in scientific_checks.items():
        if not passed:
            issues.append(issue_name)

    issues.extend(critique.issues)

    # -------------------------------------------------
    # 3. Shared plan-vs-capability checks
    # -------------------------------------------------

    issues.extend(
        _plan_capability_issues(
            plan,
            capability,
            target_variable=target_variable,
            baseline_models=baseline_models,
            candidate_models=candidate_models,
        )
    )

    issues = list(
        dict.fromkeys(issues)
    )

    if issues:
        return (
            None,
            PlanApprovalReport(
                plan_id=plan.plan_id,
                hypothesis_id=(
                    hypothesis.hypothesis_id
                ),
                passed=False,
                issues=issues,
                experiment_id=None,
            ),
        )

    # -------------------------------------------------
    # 4. Compile executable contract
    # -------------------------------------------------

    experiment_id = (
        f"EXP-{uuid4().hex[:12]}"
    )

    contract = _build_contract(
        plan,
        capability,
        target_variable=target_variable,
        baseline_models=baseline_models,
        candidate_models=candidate_models,
        experiment_id=experiment_id,
    )

    return (
        contract,
        PlanApprovalReport(
            plan_id=plan.plan_id,
            hypothesis_id=(
                hypothesis.hypothesis_id
            ),
            passed=True,
            issues=[],
            experiment_id=experiment_id,
        ),
    )
