from __future__ import annotations

from enum import StrEnum
import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from boilermind.core.contracts import (
    ExperimentAudit,
    ExperimentPlan,
    ExperimentResult,
    ScientificResult,
)
from boilermind.core.enums import ExperimentStatus, ScientificVerdict
from boilermind.planning.experiment_requirement_parser import (
    FrozenHypothesisDesign,
    frozen_design_sha256,
)


class ReflectionType(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MODEL_PERFORMANCE_ISSUE = "model_performance_issue"
    DATA_ISSUE = "data_issue"
    HYPOTHESIS_ISSUE = "hypothesis_issue"
    EXECUTION_ISSUE = "execution_issue"


class ExperimentReflection(BaseModel):
    """Deterministic post-experiment diagnosis; never changes the result."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    scientific_result: dict[str, Any]
    reflection_type: ReflectionType
    diagnosis: str = Field(min_length=1)
    recommended_changes: list[str] = Field(default_factory=list)
    next_experiment_required: bool
    next_experiment_strategy: dict[str, Any] = Field(default_factory=dict)


class ReflectionResult(BaseModel):
    """Evidence-driven hypothesis reflection without changing the verdict."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    original_hypothesis_id: str = Field(min_length=1)
    compiled_hypothesis_id: str = Field(min_length=1)
    scientific_verdict: ScientificVerdict
    observation_summary: dict[str, Any]
    hypothesis_status: str = Field(min_length=1)
    failed_claims: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    contradiction_analysis: dict[str, Any]
    next_research_direction: str = Field(min_length=1)
    next_experiment_strategy: dict[str, Any] = Field(default_factory=dict)
    successor_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)
    transition_type: str = Field(min_length=1)


_REGIME_GREATER = re.compile(
    r"all_models_regime_metric_greater:([^|:]+)\|([^|:]+)\|([A-Za-z0-9²]+)"
)


def _successor_id(hypothesis_id: str, *, reinforced: bool = False) -> str:
    if reinforced:
        return f"{hypothesis_id}-R"
    if hypothesis_id.endswith("-A"):
        return f"{hypothesis_id[:-2]}-B"
    return f"{hypothesis_id}-B"


def _regime_observation(
    result: ExperimentResult,
    criteria: list[str],
) -> dict[str, Any]:
    criterion = next(
        (item for item in criteria if _REGIME_GREATER.search(item)),
        "",
    )
    match = _REGIME_GREATER.search(criterion)
    if match is None:
        return {"criterion": criterion, "comparisons": [], "direction": "unknown"}
    left, right, metric = match.groups()
    comparisons = []
    for model_name, regimes in result.regime_metrics.items():
        left_value = regimes.get(left, {}).get(metric)
        right_value = regimes.get(right, {}).get(metric)
        if left_value is None or right_value is None:
            continue
        comparisons.append({
            "model_name": model_name,
            "left_regime": left,
            "left_value": float(left_value),
            "right_regime": right,
            "right_value": float(right_value),
            "metric": metric,
            "observed_relation": (
                "greater" if left_value > right_value
                else "less" if left_value < right_value
                else "equal"
            ),
        })
    relations = {item["observed_relation"] for item in comparisons}
    direction = next(iter(relations)) if len(relations) == 1 else "mixed"
    return {
        "criterion": criterion,
        "left_regime": left,
        "right_regime": right,
        "metric": metric,
        "comparisons": comparisons,
        "direction": direction,
    }


def _build_successor_hypothesis(
    compiled_hypothesis: dict[str, Any],
    observation: dict[str, Any],
    experiment_id: str,
    *,
    reinforced: bool,
) -> dict[str, Any]:
    parent_id = str(
        compiled_hypothesis.get("hypothesis_id")
        or compiled_hypothesis.get("id")
    )
    successor_id = _successor_id(parent_id, reinforced=reinforced)
    successor = json.loads(json.dumps(compiled_hypothesis, ensure_ascii=False))
    if reinforced:
        statement = str(
            compiled_hypothesis.get("hypothesis")
            or compiled_hypothesis.get("hypothesis_statement")
        )
        transition_type = "reinforced"
        title = "实验支持后的强化假设"
    else:
        left = str(observation["left_regime"])
        right = str(observation["right_regime"])
        metric = str(observation["metric"])
        target = next(
            iter(
                (compiled_hypothesis.get("scientific_design") or {}).get(
                    "required_targets", []
                )
            ),
            "target_variable",
        )
        condition = "、".join(
            str(item) for item in compiled_hypothesis.get(
                "applicability_conditions", []
            )
        ) or "预声明适用工况"
        statement = (
            f"在{condition}下，{right}工况的{target}预测{metric}"
            f"高于{left}工况。"
        )
        transition_type = "contradicted"
        title = "实验反向证据驱动的后继假设"

    successor.update({
        "id": successor_id,
        "hypothesis_id": successor_id,
        "title": title,
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "inference": statement,
        "expected_observation": statement,
        "verification_intent": statement,
        "workflow_status": "REFLECTION_GENERATED",
        "status": "qualified",
        "parent_hypothesis_id": parent_id,
        "source_experiment_ids": list(dict.fromkeys([
            *compiled_hypothesis.get("source_experiment_ids", []),
            experiment_id,
        ])),
        "hypothesis_evolution": {
            "parent_hypothesis_id": parent_id,
            "source_experiment_id": experiment_id,
            "transition_type": transition_type,
            "original_hypothesis_preserved": True,
        },
    })

    design_payload = successor.get("scientific_design")
    if isinstance(design_payload, dict) and not reinforced:
        design = FrozenHypothesisDesign.model_validate(design_payload)
        left = str(observation["left_regime"])
        right = str(observation["right_regime"])
        metric = str(observation["metric"])
        design = design.model_copy(update={
            "control": {"regime": left},
            "treatment": {"regime": right},
            "confirmation_criteria": [
                f"all_models_regime_metric_greater:{right}|{left}|{metric}"
            ],
            "falsification_criteria": [
                f"all_models_regime_metric_not_greater:{right}|{left}|{metric}"
            ],
        })
        successor["scientific_design"] = design.model_dump(mode="json")
        successor["scientific_design_sha256"] = frozen_design_sha256(design)
        successor["confirmation_criteria"] = list(design.confirmation_criteria)
        successor["falsification_criteria"] = list(design.falsification_criteria)
        mapping = dict(successor.get("verification_mapping") or {})
        mapping.update({
            "observable_premise": statement,
            "executable_now": True,
            "recommended_action": "EXECUTE_NOW",
        })
        successor["verification_mapping"] = mapping

    successor["raw_hypothesis_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "title": title,
                "hypothesis_statement": statement,
                "source_experiment_id": experiment_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return successor


def analyze_experiment_reflection(
    experiment_result: ExperimentResult,
    audit: ExperimentAudit,
    scientific_result: ScientificResult,
    original_hypothesis: dict[str, Any],
    compiled_hypothesis: dict[str, Any],
    experiment_criteria: dict[str, list[str]],
) -> ReflectionResult:
    """Derive hypothesis evolution only from audited experiment facts."""
    raw_model_records = experiment_result.model_records
    if isinstance(raw_model_records, dict):
        model_records = list(raw_model_records.values())
    else:
        model_records = list(raw_model_records or [])
    confirmation = list(experiment_criteria.get("confirmation_criteria", []))
    observation = _regime_observation(experiment_result, confirmation)
    original_id = str(
        original_hypothesis.get("hypothesis_id")
        or original_hypothesis.get("id")
        or compiled_hypothesis.get("original_hypothesis_id")
    )
    compiled_id = str(
        compiled_hypothesis.get("hypothesis_id")
        or compiled_hypothesis.get("id")
    )
    identity = {
        "experiment_id": experiment_result.experiment_id,
        "original_hypothesis_id": original_id,
        "compiled_hypothesis_id": compiled_id,
        "scientific_verdict": scientific_result.verdict,
        "observation_summary": observation,
    }
    if not audit.execution_valid:
        return ReflectionResult(
            **identity,
            hypothesis_status="experiment_design_insufficient",
            failed_claims=[],
            supported_claims=[],
            contradiction_analysis={
                "detected": False,
                "reason": "experiment_audit_not_valid",
            },
            next_research_direction="修复实验执行或审计问题后重新冻结设计。",
            next_experiment_strategy={"strategy": "repair_execution_or_audit"},
            successor_hypotheses=[],
            reasoning="ExperimentAudit未通过，实验事实不得驱动假设方向变化。",
            transition_type="design_revision_required",
        )

    compiled_claim = str(
        compiled_hypothesis.get("hypothesis")
        or compiled_hypothesis.get("hypothesis_statement")
    )
    original_claim = str(
        original_hypothesis.get("hypothesis")
        or original_hypothesis.get("hypothesis_statement")
    )
    if scientific_result.verdict == ScientificVerdict.SUPPORTED:
        successor = _build_successor_hypothesis(
            compiled_hypothesis,
            observation,
            experiment_result.experiment_id,
            reinforced=True,
        )
        return ReflectionResult(
            **identity,
            hypothesis_status="supported",
            failed_claims=[],
            supported_claims=[original_claim or compiled_claim],
            contradiction_analysis={"detected": False, "comparisons": []},
            next_research_direction="保留原假设并形成带实验溯源的强化假设。",
            next_experiment_strategy={"strategy": "stop_resolved"},
            successor_hypotheses=[successor],
            reasoning="有效实验的ScientificResult明确为supported。",
            transition_type="reinforced",
        )

    contradiction = bool(observation["comparisons"]) and (
        observation["direction"] == "less"
    )
    if contradiction:
        successor = _build_successor_hypothesis(
            compiled_hypothesis,
            observation,
            experiment_result.experiment_id,
            reinforced=False,
        )
        return ReflectionResult(
            **identity,
            hypothesis_status="observable_premise_falsified",
            failed_claims=[compiled_claim],
            supported_claims=[successor["hypothesis"]],
            contradiction_analysis={
                "detected": True,
                "expected_relation": "greater",
                "observed_relation": observation["direction"],
                "comparisons": observation["comparisons"],
            },
            next_research_direction="验证实验观察到的反向工况误差关系。",
            next_experiment_strategy={
                "strategy": "test_successor_hypothesis",
                "successor_hypothesis_id": successor["hypothesis_id"],
            },
            successor_hypotheses=[successor],
            reasoning=(
                "所有具有完整分层指标的已执行模型均观察到预声明方向的反向关系；"
                "仅生成反事实后继假设，不改写原ScientificResult。"
            ),
            transition_type="contradicted",
        )

    diagnosed_gaps: list[str] = []
    if not model_records:
        diagnosed_gaps.append("missing_executed_models")
    if not experiment_result.regime_metrics:
        diagnosed_gaps.append("missing_regime_coverage")
    if any(
        not record.fit_success for record in model_records
    ):
        diagnosed_gaps.append("model_execution_failure")
    unsupported = list(compiled_hypothesis.get("unsupported_extensions", []))
    if unsupported:
        diagnosed_gaps.append("unsupported_hypothesis_extensions")
    if not diagnosed_gaps:
        diagnosed_gaps.append("criteria_not_discriminated_by_current_observation")
    return ReflectionResult(
        **identity,
        hypothesis_status="open_insufficient_evidence",
        failed_claims=list(scientific_result.failed_criteria),
        supported_claims=list(scientific_result.achieved_criteria),
        contradiction_analysis={
            "detected": False,
            "reason": "no_consistent_reverse_observation",
        },
        next_research_direction="保持假设开放，补充能够区分确认与证伪条件的实验设计。",
        next_experiment_strategy={
            "strategy": "refine_experiment_design",
            "diagnosed_gaps": diagnosed_gaps,
        },
        successor_hypotheses=[],
        reasoning="当前结果不足以确定支持或反向关系，禁止自动生成方向性结论。",
        transition_type="refined",
    )


def reflect_experiment(
    experiment_result: ExperimentResult,
    audit: ExperimentAudit,
    scientific_result: ScientificResult,
) -> ExperimentReflection:
    """Classify an audited outcome using only existing deterministic facts."""
    scientific_snapshot = scientific_result.model_dump(mode="json")

    if not audit.execution_valid or experiment_result.status in {
        ExperimentStatus.FAILED,
        ExperimentStatus.INVALID,
    }:
        return ExperimentReflection(
            experiment_id=experiment_result.experiment_id,
            scientific_result=scientific_snapshot,
            reflection_type=ReflectionType.EXECUTION_ISSUE,
            diagnosis="实验执行或审计未通过，不能据此安排自动科学迭代。",
            recommended_changes=["修复审计或执行问题后重新冻结实验合同"],
            next_experiment_required=False,
        )

    if scientific_result.verdict in {
        ScientificVerdict.SUPPORTED,
        ScientificVerdict.FALSIFIED,
    }:
        return ExperimentReflection(
            experiment_id=experiment_result.experiment_id,
            scientific_result=scientific_snapshot,
            reflection_type=ReflectionType.INSUFFICIENT_EVIDENCE,
            diagnosis="科学判定已形成，不需要自动追加实验。",
            recommended_changes=[],
            next_experiment_required=False,
            next_experiment_strategy={"strategy": "stop_resolved"},
        )

    issue_text = " ".join(
        [
            *experiment_result.execution_notes,
            *experiment_result.experiment_validity_issues,
            *scientific_result.failed_criteria,
            scientific_result.rationale,
        ]
    ).casefold()
    if any(token in issue_text for token in ("sample", "样本", "coverage", "覆盖不足")):
        reflection_type = ReflectionType.DATA_ISSUE
        diagnosis = "实验有效，但样本数量或工况覆盖不足以形成稳定科学判定。"
    elif any(token in issue_text for token in ("model", "模型", "metric_not", "性能")):
        reflection_type = ReflectionType.MODEL_PERFORMANCE_ISSUE
        diagnosis = "实验有效，但当前模型集合的观测未达到预先冻结的判定标准。"
    elif scientific_result.failed_criteria:
        reflection_type = ReflectionType.HYPOTHESIS_ISSUE
        diagnosis = "实验有效，但当前实验设计不能充分区分假设的确认与证伪条件。"
    else:
        reflection_type = ReflectionType.INSUFFICIENT_EVIDENCE
        diagnosis = "实验完成，但当前实验条件不足以支持或证伪假设。"

    return ExperimentReflection(
        experiment_id=experiment_result.experiment_id,
        scientific_result=scientific_snapshot,
        reflection_type=reflection_type,
        diagnosis=diagnosis,
        recommended_changes=[
            "优先扩展当前CapabilityRegistry确认可执行的模型集合",
            "若能力允许，再调整训练窗口或显式特征集合",
            "若无安全设计扩展，则使用新的随机种子进行独立重复验证",
        ],
        next_experiment_required=True,
        next_experiment_strategy={"strategy": "pending_plan_compilation"},
    )


def build_reflected_plan(
    current_plan: ExperimentPlan,
    reflection: ExperimentReflection,
    *,
    available_models: list[str] | tuple[str, ...] = (),
    allowed_window_steps: list[int] | tuple[int, ...] = (),
    round_index: int = 2,
) -> ExperimentPlan | None:
    """Build a plan variant without changing frozen evaluation semantics."""
    if not reflection.next_experiment_required:
        return None

    current_models = list(current_plan.candidate_models)
    additions = [
        model for model in available_models
        if model not in current_models and model not in current_plan.reference_models
    ]
    update: dict[str, Any] = {
        "plan_id": f"{current_plan.plan_id}-R{round_index}",
        "status": "planned",
    }
    strategy: dict[str, Any]

    if additions:
        expanded = current_models + additions[:2]
        update.update({
            "candidate_models": expanded,
            "model_candidates": expanded,
            "recommended_models": list(dict.fromkeys(
                [*current_plan.recommended_models, *additions[:2]]
            )),
            "executable_models": list(dict.fromkeys(
                [*current_plan.executable_models, *additions[:2]]
            )),
            "model_substitution_reason": (
                current_plan.model_substitution_reason
                or "experiment_reflection_added_registry_executable_models"
            ),
        })
        strategy = {
            "strategy": "expand_executable_model_set",
            "added_models": additions[:2],
        }
    else:
        larger_windows = sorted({
            int(value) for value in allowed_window_steps
            if int(value) > current_plan.window_steps
        })
        if larger_windows:
            update["window_steps"] = larger_windows[0]
            strategy = {
                "strategy": "increase_window_steps",
                "from": current_plan.window_steps,
                "to": larger_windows[0],
            }
        else:
            update["random_seed"] = current_plan.random_seed + round_index - 1
            strategy = {
                "strategy": "independent_seed_replication",
                "from": current_plan.random_seed,
                "to": update["random_seed"],
            }

    # This provenance changes neither locked-test isolation nor metric/target.
    requirements = dict(current_plan.execution_requirements)
    requirements["experiment_reflection"] = {
        "source_experiment_id": reflection.experiment_id,
        "reflection_type": reflection.reflection_type.value,
        **strategy,
    }
    update["execution_requirements"] = requirements
    reflection.next_experiment_strategy = strategy
    return current_plan.model_copy(update=update)
