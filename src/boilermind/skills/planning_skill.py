from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Any

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.models import (
    build_default_registry,
)

from boilermind.planning.experiment_requirement_parser import (
    EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON,
    FrozenHypothesisDesign,
    frozen_design_sha256,
    parse_hypothesis_requirements,
    requirements_from_frozen_design,
)
from boilermind.planning.parameter_optimization import expand_parameter_plans

from .base import BaseSkill
from .model_selection_skill import ModelSelectionSkill


_COMPUTE_COST_RANK = {
    "trivial": 0,
    "seconds": 1,
    "minutes": 2,
    "hours": 3,
}


def _canonical_metric_names(values: list[Any]) -> list[str]:
    """Extract only metrics supported by the experiment contract."""
    supported = ("MAE", "RMSE", "R2", "MBE")
    found: list[str] = []
    for value in values:
        text = str(value).upper()
        for metric in supported:
            if re.search(rf"(?<![A-Z0-9_]){metric}(?![A-Z0-9_])", text):
                found.append(metric)
    return list(dict.fromkeys(found))


def _critical_text_corrupted(hypothesis: dict[str, Any]) -> bool:
    text = " ".join(
        str(hypothesis.get(key, ""))
        for key in (
            "title", "hypothesis", "statement", "mechanism",
            "verification_intent", "falsification_condition",
        )
    )
    if "\ufffd" in text:
        return True
    mojibake_markers = ("锛", "銆", "鈥", "璐熻嵎", "棰勬祴")
    return "�" in text or sum(marker in text for marker in mojibake_markers) >= 2


class PlanningSkill(BaseSkill):
    """
    Selected-hypothesis-driven experiment planner.

    Flow (fail closed):

    selected_hypothesis_id
      -> real QualifiedHypothesis
      -> ExperimentRequirements (deterministic parser)
      -> CapabilityRegistry.check_executable
      -> ModelRegistry.compatible_with_capability
      -> executable_model_pool
      -> planner selects candidate / reference models
      -> ExperimentPlan

    No hard-coded model candidates.
    No H001 fallback.
    No fake experiment when capability is missing.
    """

    name = "experiment_planning"
    description = "根据真实科研假设生成可执行实验规划"

    def __init__(
        self,
        *,
        capability_registry: (
            ExperimentCapabilityRegistry | None
        ) = None,
        model_registry: Any | None = None,
    ):
        self.capability = (
            capability_registry
            or ExperimentCapabilityRegistry()
        )

        self.model_registry = (
            model_registry
            or build_default_registry()
        )
        self.model_selection = ModelSelectionSkill(
            model_registry=self.model_registry,
            capability_registry=self.capability,
        )

    # ---------------------------------------------------------
    # Hypothesis resolution (strict, no fallback)
    # ---------------------------------------------------------

    def _resolve_selected_hypothesis(
        self,
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:

        selected_hypothesis_id = context.get(
            "selected_hypothesis_id"
        )

        if not selected_hypothesis_id:
            raise ValueError(
                "selected_hypothesis_id_required"
            )

        hypotheses = (
            context.get("qualified_hypotheses")
            or context.get("hypotheses")
            or []
        )

        if not isinstance(hypotheses, list):
            raise ValueError(
                "hypotheses_list_required"
            )

        selected = None

        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue

            hypothesis_id = (
                hypothesis.get("hypothesis_id")
                or hypothesis.get("id")
            )

            if (
                hypothesis_id
                == selected_hypothesis_id
            ):
                selected = hypothesis
                break

        if selected is None:
            raise ValueError(
                "selected_hypothesis_not_found:"
                f"{selected_hypothesis_id}"
            )

        return str(
            selected_hypothesis_id
        ), selected

    # ---------------------------------------------------------
    # Capability match + model pool
    # ---------------------------------------------------------

    @staticmethod
    def _task_type(requirements, hypothesis: dict[str, Any]) -> str:
        text = " ".join(
            str(hypothesis.get(key, ""))
            for key in ("hypothesis", "verification_intent", "research_goal")
        ).casefold()
        if requirements.experiment_type == "constrained_optimization" or any(
            token in text for token in ("优化", "optimization", "给煤", "送风")
        ):
            return "optimization"
        if any(token in text for token in ("故障", "诊断", "diagnos", "fault")):
            return "diagnosis"
        return "prediction"

    def _select_models(
        self,
        requirements,
        *,
        target: str,
        hypothesis: dict[str, Any],
        objective: str = "",
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:

        selection = self.model_selection.execute({
            "hypothesis": hypothesis,
            "task_type": self._task_type(requirements, hypothesis),
            "target_variable": target,
            "operating_condition": str(
                hypothesis.get("applicability_conditions")
                or hypothesis.get("operating_condition")
                or ""
            ),
            "metrics": metrics or requirements.required_metrics,
            "required_models": requirements.required_models,
            "objective": objective or str(
                hypothesis.get("verification_intent")
                or hypothesis.get("hypothesis")
                or ""
            ),
        })
        return selection

    def _select_representative_candidates(
        self,
        pool,
        *,
        max_candidates: int = 3,
    ) -> list[str]:
        """
        Deterministic cost-aware selection:

        1. sort executable models by ModelRegistry
           compute_cost (trivial < seconds < minutes <
           hours), then by model name;
        2. greedily keep one model per family until
           max_candidates is reached;
        3. fill any remaining slots by cost order.
        """

        ordered = sorted(
            pool,
            key=lambda spec: (
                _COMPUTE_COST_RANK.get(
                    spec.compute_cost,
                    9,
                ),
                spec.model_name,
            ),
        )

        chosen: list[str] = []
        seen_families: set[str] = set()

        for spec in ordered:
            family = (
                spec.family
                or spec.framework
            )

            if family in seen_families:
                continue

            chosen.append(spec.model_name)
            seen_families.add(family)

            if len(chosen) >= max_candidates:
                break

        if len(chosen) < max_candidates:
            for spec in ordered:
                if spec.model_name in chosen:
                    continue

                chosen.append(spec.model_name)

                if len(chosen) >= max_candidates:
                    break

        return sorted(chosen)

    # ---------------------------------------------------------
    # Plan assembly
    # ---------------------------------------------------------

    def _build_plan(
        self,
        *,
        hypothesis_id: str,
        problem_id: str | None,
        hypothesis: dict[str, Any],
        requirements,
        candidate_models: list[str],
        reference_models: list[str],
        recommended_models: list[str],
        executable_models: list[str],
        model_substitution_reason: str,
        rationale: str,
        target: str,
        target_inference_reason: str = "",
        evaluation_objective: str = "",
        evaluation_metrics: list[str] | None = None,
    ) -> dict[str, Any]:

        statement = str(
            hypothesis.get("hypothesis")
            or hypothesis.get("statement")
            or ""
        )

        horizon_steps = (
            requirements.prediction_horizon_steps
            or self.capability.prediction_horizon_steps
        )

        frozen_metrics = list(evaluation_metrics or requirements.required_metrics)
        primary_metric = (
            frozen_metrics[0]
            if frozen_metrics
            else "MAE"
        )

        secondary_metrics = frozen_metrics[1:]

        selection_metric = (
            f"validation_{primary_metric.lower()}"
            f"_t_h"
            if primary_metric in {"MAE", "RMSE", "MBE"}
            else (
                f"validation_{primary_metric.lower()}"
            )
        )

        control = dict(requirements.control)
        treatment = dict(requirements.treatment)
        expected_observation = str(
            hypothesis.get("verification_mapping", {}).get("observable_premise")
            or hypothesis.get("expected_observation")
            or hypothesis.get("inference")
            or (
                requirements.confirmation_criteria[0]
                if requirements.confirmation_criteria else ""
            )
            or ""
        ).strip()
        immutable_hash = str(
            hypothesis.get("raw_hypothesis_sha256", "")
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", immutable_hash):
            immutable_hash = hashlib.sha256(
                json.dumps(
                    {
                        "title": hypothesis.get("title", ""),
                        "hypothesis_statement": statement,
                        "engineering_mechanism": hypothesis.get("mechanism", ""),
                        "expected_observation": expected_observation,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        measurement = (
            f"operations={','.join(requirements.required_operations)};"
            f"metrics={','.join(frozen_metrics)};target={target}"
        )
        hypothesis_binding = {
            "hypothesis_id": hypothesis_id,
            "immutable_hypothesis_sha256": immutable_hash,
            "expected_observation": expected_observation,
            "experiment_measurement": measurement,
            "confirmation_mapping": list(requirements.confirmation_criteria),
            "falsification_mapping": list(requirements.falsification_criteria),
        }
        verification_mapping = hypothesis.get("verification_mapping", {})
        if isinstance(verification_mapping, dict) and verification_mapping.get(
            "verification_scope"
        ):
            hypothesis_binding["verification_scope"] = verification_mapping[
                "verification_scope"
            ]
            hypothesis_binding["deferred_missing_capabilities"] = list(
                verification_mapping.get("deferred_missing_capabilities", [])
            )

        return {
            # ---- hypothesis linkage (P0-3 core) ----
            "plan_id": f"PLAN-{hypothesis_id}",
            "problem_id": problem_id,
            "hypothesis_id": hypothesis_id,
            "hypothesis_statement": statement,
            "hypothesis_binding": hypothesis_binding,

            # ---- experiment semantics ----
            "experiment_type": (
                requirements.experiment_type
            ),
            "required_operations": list(
                requirements.required_operations
            ),

            "candidate_models": list(
                candidate_models
            ),
            "reference_models": list(
                reference_models
            ),
            "recommended_models": list(recommended_models),
            "executable_models": list(executable_models),
            "model_substitution_reason": model_substitution_reason,

            "control": control,
            "treatment": treatment,

            "target": target,
            "target_inference_reason": target_inference_reason,
            "prediction_horizon_steps": int(
                horizon_steps
            ),

            "primary_metric": primary_metric,
            "secondary_metrics": list(
                secondary_metrics
            ),

            "hard_constraints": list(
                requirements.hard_constraints
            ),

            "confirmation_criteria": list(
                requirements.confirmation_criteria
            ),
            "falsification_criteria": list(
                requirements.falsification_criteria
            ),

            "current_executable": True,
            "missing_capabilities": [],
            "model_selection_rationale": rationale,

            # ---- runtime contract (real backend) ----
            "execution_backend": "real_sklearn",
            "dataset_path": str(
                self.capability.dataset_path.resolve()
            ),
            "execution_requirements": {
                "dataset_path": str(
                    self.capability.dataset_path.resolve()
                ),
            },
            "sampling_interval_seconds": (
                self.capability.sampling_interval_seconds
            ),
            "window_steps": self.capability.window_steps,
            "train_ratio": self.capability.train_ratio,
            "validation_ratio": (
                self.capability.validation_ratio
            ),
            "required_variables": (
                self.capability.available_variables()
            ),
            "metrics": list(
                frozen_metrics
            ),
            "model_candidates": list(
                candidate_models
            ),
            "reference_model": (
                reference_models[0]
                if reference_models
                else None
            ),
            "selection_metric": selection_metric,
            "locked_test_used_for_selection": False,
            "random_seed": 42,
            "output_dir": os.environ.get(
                "BOILERMIND_EXPERIMENT_OUTPUT_DIR",
                (
                    r"D:\BoilerMind-Trusted"
                    r"\outputs\experiments"
                ),
            ),

            # ---- legacy ExperimentPlan fields ----
            "objective": str(
                evaluation_objective
                or treatment.get("description")
                or statement
            ),
            "experimental_design": (
                f"{requirements.experiment_type};"
                f"operations="
                f"{','.join(requirements.required_operations)}"
            ),
            "baseline_description": str(
                control.get("description")
                or "无对照"
            ),
            "intervention_description": str(
                treatment.get("description")
                or "无干预"
            ),
            "expected_observation": (
                requirements.confirmation_criteria[0]
                if requirements.confirmation_criteria
                else ""
            ),

            "status": "planned",
        }

    def _not_executable_result(
        self,
        *,
        hypothesis_id: str,
        problem_id: str | None,
        hypothesis: dict[str, Any],
        requirements,
        missing_capabilities: list[str],
    ) -> dict[str, Any]:

        report = {
            "plan_id": f"PLAN-{hypothesis_id}",
            "problem_id": problem_id,
            "hypothesis_id": hypothesis_id,
            "hypothesis_statement": str(
                hypothesis.get("hypothesis")
                or hypothesis.get("statement")
                or ""
            ),
            "experiment_type": (
                requirements.experiment_type
            ),
            "required_operations": list(
                requirements.required_operations
            ),
            "candidate_models": [],
            "reference_models": [],
            "control": dict(requirements.control),
            "treatment": dict(requirements.treatment),
            "target": (
                requirements.required_targets[0]
                if requirements.required_targets
                else None
            ),
            "prediction_horizon_steps": (
                requirements.prediction_horizon_steps
            ),
            "primary_metric": (
                requirements.required_metrics[0]
                if requirements.required_metrics
                else None
            ),
            "secondary_metrics": list(
                requirements.required_metrics[1:]
            ),
            "hard_constraints": list(
                requirements.hard_constraints
            ),
            "confirmation_criteria": list(
                requirements.confirmation_criteria
            ),
            "falsification_criteria": list(
                requirements.falsification_criteria
            ),
            "current_executable": False,
            "missing_capabilities": list(
                dict.fromkeys(
                    missing_capabilities
                )
            ),
            "model_selection_rationale": (
                "planning_aborted_capability_mismatch"
            ),
            "status": "not_executable",
        }

        return {
            "experiment_plan": None,
            "planning_report": report,
            "current_executable": False,
            "missing_capabilities": report[
                "missing_capabilities"
            ],
            "status": "not_executable",
        }

    # ---------------------------------------------------------
    # Skill entry
    # ---------------------------------------------------------

    def execute(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        hypothesis_id, hypothesis = (
            self._resolve_selected_hypothesis(
                context
            )
        )

        research_problem = context.get(
            "research_problem",
            {},
        )

        problem_id = (
            context.get("problem_id")
            or (
                research_problem.get("problem_id")
                if isinstance(
                    research_problem,
                    dict,
                )
                else None
            )
        )

        if not problem_id:
            raise ValueError(
                "problem_id_required"
            )

        frozen_payload = hypothesis.get("scientific_design")
        frozen_hash = str(
            hypothesis.get("scientific_design_sha256") or ""
        )
        if frozen_payload is not None:
            try:
                frozen_design = FrozenHypothesisDesign.model_validate(
                    frozen_payload
                )
                actual_hash = frozen_design_sha256(frozen_design)
            except Exception as exc:
                frozen_design = None
                actual_hash = ""
                design_issues = [f"hypothesis_design:invalid:{exc}"]
            else:
                design_issues = (
                    [] if frozen_hash == actual_hash
                    else ["hypothesis_design:sha256_mismatch"]
                )
            requirements = (
                requirements_from_frozen_design(hypothesis_id, frozen_design)
                if frozen_design is not None
                else parse_hypothesis_requirements(hypothesis)
            )
        else:
            # Compatibility path for imported historical hypotheses only.
            requirements = parse_hypothesis_requirements(hypothesis)
            design_issues = []

        if design_issues:
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=design_issues,
            )

        verification_mapping = hypothesis.get("verification_mapping", {})
        if isinstance(verification_mapping, dict) and not verification_mapping.get(
            "executable_now", True
        ):
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=(
                    list(verification_mapping.get("missing_capabilities", []))
                    + [
                        f"user_input:{item}"
                        for item in verification_mapping.get("required_user_inputs", [])
                    ]
                ) or ["hypothesis_not_mapped_to_executable_operation"],
            )

        staged_observable_premise = (
            isinstance(verification_mapping, dict)
            and verification_mapping.get("verification_scope")
            == "problem_observable_premise_only"
        )
        if staged_observable_premise:
            requirements = requirements.model_copy(update={
                "experiment_type": "regime_stratified_evaluation",
                "required_operations": ["regime_stratified_evaluation"],
                "required_models": [],
                "required_model_roles": {},
                "required_metrics": ["MAE"],
                "required_variables": [],
                "control": {"regime": "ramp_down"},
                "treatment": {"regime": "ramp_up"},
                "confirmation_criteria": [
                    "all_models_regime_metric_greater:ramp_up|ramp_down|MAE"
                ],
                "falsification_criteria": [
                    "all_models_regime_metric_not_greater:ramp_up|ramp_down|MAE"
                ],
                "requires_feature_intervention": False,
            })

        if _critical_text_corrupted(hypothesis):
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=[
                    "text_encoding:critical_hypothesis_field_corrupted"
                ],
            )

        if requirements.hypothesis_id != hypothesis_id:
            raise ValueError(
                "hypothesis_id_mismatch_in_parser"
            )

        problem_payload = research_problem if isinstance(research_problem, dict) else {}
        problem_models = [] if frozen_payload is not None else [
            str(item).strip().lower()
            for item in (
                list(problem_payload.get("required_models", []))
                + list(problem_payload.get("reference_models", []))
            )
            if str(item).strip()
        ]
        problem_operations = [
            str(item).strip()
            for item in problem_payload.get("required_operations", [])
            if str(item).strip()
        ]
        problem_constraints = [
            str(item).strip()
            for item in problem_payload.get("protocol_constraints", [])
            if str(item).strip()
        ]
        problem_horizon = problem_payload.get("required_horizon_steps")
        hypothesis_horizon = requirements.prediction_horizon_steps
        if (
            problem_horizon is not None
            and hypothesis_horizon is not None
            and int(problem_horizon) != int(hypothesis_horizon)
        ):
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=[
                    "protocol:prediction_horizon_conflict:"
                    f"problem={int(problem_horizon)}:"
                    f"hypothesis={int(hypothesis_horizon)}"
                ],
            )
        merged_roles = dict(requirements.required_model_roles)
        for model in problem_models:
            merged_roles[model] = (
                "reference"
                if model in {
                    str(item).strip().lower()
                    for item in problem_payload.get("reference_models", [])
                }
                else merged_roles.get(model, "candidate")
            )
        requirements = requirements.model_copy(
            update={
                "required_models": list(dict.fromkeys(
                    list(requirements.required_models) + problem_models
                )),
                "required_operations": sorted(set(
                    list(requirements.required_operations) + problem_operations
                )),
                "required_model_roles": merged_roles,
                "hard_constraints": list(dict.fromkeys(
                    list(requirements.hard_constraints) + problem_constraints
                )),
                "prediction_horizon_steps": (
                    int(problem_horizon)
                    if problem_horizon is not None
                    else requirements.prediction_horizon_steps
                ),
            }
        )

        match = self.capability.check_executable(
            required_operations=(
                requirements.required_operations
            ),
            required_models=(
                requirements.required_models
            ),
            required_metrics=(
                requirements.required_metrics
            ),
            requires_feature_intervention=(
                requirements.requires_feature_intervention
            ),
            required_variables=(
                requirements.required_variables
            ),
        )

        if not match.executable:
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=(
                    match.missing_capabilities
                ),
            )

        target = str(problem_payload.get("target_variable", "")).strip()
        if (
            target.casefold() in {"", "unspecified", "unknown", "none", "null"}
            and len(requirements.required_targets) == 1
        ):
            target = str(requirements.required_targets[0]).strip()
        if target.casefold() in {"", "unspecified", "unknown", "none", "null"}:
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=["target_variable_resolution_required"],
            )
        evaluation_objective = str(problem_payload.get("objective", "")).strip()
        if not evaluation_objective:
            evaluation_objective = str(
                hypothesis.get("verification_intent")
                or hypothesis.get("hypothesis")
                or ""
            )
        problem_metrics = problem_payload.get("metrics")
        selection_metrics = _canonical_metric_names(
            problem_metrics if isinstance(problem_metrics, list) else []
        ) or list(requirements.required_metrics)

        parameter_optimization_task = (
            problem_payload.get("research_task_type") == "parameter_optimization"
        )
        if (
            frozen_payload is not None
            and not staged_observable_premise
            and not parameter_optimization_task
        ):
            candidate_models = sorted(
                model for model, role
                in requirements.required_model_roles.items()
                if role != "reference"
            )
            reference_models = sorted(
                model for model, role
                in requirements.required_model_roles.items()
                if role == "reference"
            )
            selection = {
                "candidate_models": candidate_models,
                "baseline_models": reference_models,
                "recommended_models": candidate_models,
                "executable_models": candidate_models,
                "model_substitution_reason": (
                    "Frozen scientific design used without substitution."
                ),
                "selection_reason": (
                    "Models and roles come from the frozen scientific design."
                ),
                "missing_models": [],
            }
        else:
            selection = self._select_models(
                requirements,
                target=target,
                hypothesis=hypothesis,
                objective=evaluation_objective,
                metrics=selection_metrics,
            )
        if staged_observable_premise:
            available = set(self.capability.available_models())
            cheap = [
                model for model in ("ridge", "bayesianridge", "pls")
                if model in available
            ]
            if cheap:
                selection = {
                    "candidate_models": cheap[:3],
                    "baseline_models": [self.capability.reference_model_id()],
                    "recommended_models": cheap[:3],
                    "executable_models": cheap[:3],
                    "model_substitution_reason": "",
                    "selection_reason": (
                        "Observable-premise screening uses three low-cost "
                        "registered models; mechanism testing is deferred."
                    ),
                    "missing_models": [],
                }
        candidate_models = list(selection.get("candidate_models", []))
        reference_models = list(selection.get("baseline_models", []))
        recommended_models = list(selection.get("recommended_models", []))
        executable_models = list(selection.get("executable_models", []))
        substitution_reason = str(selection.get("model_substitution_reason", ""))
        rationale = str(selection.get("selection_reason", "model_selection_failed_closed"))
        missing_models = list(selection.get("missing_models", []))

        if missing_models:
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=[
                    f"model:{model_id}"
                    for model_id in missing_models
                ],
            )

        frozen_reference_models = set(requirements.required_model_roles)
        frozen_reference_models = {
            model for model in frozen_reference_models
            if requirements.required_model_roles.get(model) == "reference"
        }
        frozen_candidate_models = (
            set(requirements.required_models) - frozen_reference_models
        )
        omitted_frozen_models = sorted(
            frozen_candidate_models - set(candidate_models)
        )
        if omitted_frozen_models:
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=[
                    "plan:model_set_not_preserved:" + model_id
                    for model_id in omitted_frozen_models
                ],
            )

        if not candidate_models:
            return self._not_executable_result(
                hypothesis_id=hypothesis_id,
                problem_id=problem_id,
                hypothesis=hypothesis,
                requirements=requirements,
                missing_capabilities=[
                    "model:candidate_pool_empty"
                ],
            )

        plan = self._build_plan(
            hypothesis_id=hypothesis_id,
            problem_id=problem_id,
            hypothesis=hypothesis,
            requirements=requirements,
            candidate_models=candidate_models,
            reference_models=reference_models,
            recommended_models=recommended_models,
            executable_models=executable_models,
            model_substitution_reason=substitution_reason,
            rationale=rationale,
            target=target,
            target_inference_reason=str(
                problem_payload.get("target_inference_reason", "")
            ),
            evaluation_objective=evaluation_objective,
            evaluation_metrics=selection_metrics,
        )

        experiment_plans = [plan]
        if problem_payload.get("research_task_type") == "parameter_optimization":
            variable = str(problem_payload.get("optimization_variable") or "")
            candidates = list(problem_payload.get("candidate_values") or [])
            try:
                experiment_plans = expand_parameter_plans(
                    plan, variable=variable, candidates=candidates
                )
            except ValueError as exc:
                return self._not_executable_result(
                    hypothesis_id=hypothesis_id,
                    problem_id=problem_id,
                    hypothesis=hypothesis,
                    requirements=requirements,
                    missing_capabilities=[f"parameter_optimization:{exc}"],
                )
            if hasattr(self.capability, "supports_window_steps") and variable == "window_steps":
                unsupported = [
                    value for value in candidates
                    if not self.capability.supports_window_steps(int(value))
                ]
                if unsupported:
                    return self._not_executable_result(
                        hypothesis_id=hypothesis_id,
                        problem_id=problem_id,
                        hypothesis=hypothesis,
                        requirements=requirements,
                        missing_capabilities=[
                            "window_steps:" + ",".join(map(str, unsupported))
                        ],
                    )
            plan = experiment_plans[0]

        pool = self.model_registry.compatible_with_capability(
            self.capability,
            tasks=(
                ["steam_volume_forecast"]
                if target == "steam_volumetric_flow"
                else ["mass_flow_forecast"]
            ),
            target=target,
            metrics=requirements.required_metrics,
        )

        executable_model_pool = sorted(
            spec.model_name
            for spec in pool
        )

        selected_models = sorted(
            set(candidate_models)
            | set(reference_models)
        )

        return {
            "experiment_plan": plan,
            "experiment_plans": experiment_plans,
            "experiment_optimization_design": (
                {
                    "optimized_variable": problem_payload.get("optimization_variable"),
                    "candidates": list(problem_payload.get("candidate_values") or []),
                    "selection_metric": plan.get("selection_metric") or "MAE",
                }
                if len(experiment_plans) > 1 else None
            ),
            "planning_report": plan,
            "current_executable": True,
            "missing_capabilities": [],
            "executable_model_pool": (
                executable_model_pool
            ),
            "selected_models": selected_models,
            "status": "planned",
        }
