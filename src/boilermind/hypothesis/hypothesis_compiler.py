from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from boilermind.hypothesis.deterministic_admission import evaluate_candidate
from boilermind.planning.experiment_requirement_parser import (
    FrozenHypothesisDesign,
    frozen_design_sha256,
)


REGIME_OPERATION = "regime_stratified_evaluation"
PARAMETER_OPERATION = "model_comparison"


def _is_model_comparison_problem(problem: dict[str, Any]) -> bool:
    text = " ".join(str(problem.get(key, "")) for key in (
        "original_question", "objective", "research_goal",
    )).casefold()
    explicit_operation = "model_comparison" in {
        str(item).strip().casefold()
        for item in problem.get("required_operations", [])
    }
    return explicit_operation or any(token in text for token in (
        "哪种模型", "哪个模型", "模型最好", "最佳模型",
        "模型比较", "模型对比", "模型选择", "比较不同模型",
        "which model", "best model", "model comparison",
    ))


def _model_comparison_candidate(
    originals: list[dict[str, Any]],
    problem: dict[str, Any],
    capability_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], CompiledHypothesis]:
    original = originals[0]
    hypothesis_id = str(original.get("hypothesis_id") or original.get("id"))
    target = str(problem.get("target_variable") or "目标变量")
    condition = str(problem.get("operating_condition") or "预声明运行范围")
    capability_models = list(dict.fromkeys(
        str(item).strip().casefold()
        for item in (
            capability_snapshot.get("enabled_experiment_models", [])
            or capability_snapshot.get("models", [])
        )
        if str(item).strip()
    ))
    requested_models = list(dict.fromkeys(
        str(item).strip().casefold()
        for item in problem.get("required_models", [])
        if str(item).strip()
    ))
    models = requested_models or capability_models
    requested_references = [
        str(item).strip().casefold()
        for item in problem.get("reference_models", [])
        if str(item).strip()
    ]
    reference = str(
        requested_references[0]
        if requested_references
        else capability_snapshot.get("reference_model") or ""
    ).strip().casefold()
    models = [item for item in models if item != reference]
    available_metrics = {
        str(item).strip().upper()
        for item in capability_snapshot.get("available_metrics", [])
    }
    requested_metrics = [
        str(item).strip().upper() for item in problem.get("metrics", [])
        if str(item).strip().upper() in available_metrics
    ]
    metrics = list(dict.fromkeys(requested_metrics or ["MAE", "RMSE", "R2"]))
    primary = metrics[0]
    model_csv = ",".join(models)
    statement = (
        f"在{condition}下，当前注册候选模型对{target}的validation性能"
        f"存在可比较差异，可依据冻结主指标{primary}选择最佳模型。"
    )
    confirmation = [
        f"any_model_better_than_model_on:{model_csv}|{reference}|{primary}"
    ] if reference else [f"best_model_identifiable_on:{model_csv}|{primary}"]
    falsification = [
        f"all_models_not_better_than_model_on:{model_csv}|{reference}|{primary}"
    ] if reference else [f"best_model_not_identifiable_on:{model_csv}|{primary}"]
    required_operations = ["model_comparison", "chronological_validation"]
    if reference:
        required_operations.append("reference_model_comparison")
    if capability_snapshot.get("locked_test_supported"):
        required_operations.append("locked_test_evaluation")
    roles = {item: "candidate" for item in models}
    if reference:
        roles[reference] = "reference"
    horizon = (
        problem.get("required_horizon_steps")
        or capability_snapshot.get("prediction_horizon_steps")
    )
    design = FrozenHypothesisDesign(
        experiment_type=("reference_model_comparison" if reference else "model_comparison"),
        required_operations=required_operations,
        required_models=models + ([reference] if reference else []),
        required_model_roles=roles,
        required_targets=[target],
        required_metrics=metrics,
        prediction_horizon_steps=horizon,
        control={"models": [reference]} if reference else {},
        treatment={"models": models},
        confirmation_criteria=confirmation,
        falsification_criteria=falsification,
    )
    candidate = deepcopy(original)
    candidate.update({
        "id": hypothesis_id,
        "hypothesis_id": hypothesis_id,
        "title": "当前注册模型公平比较",
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "mechanism": "所有候选模型使用同一冻结数据、切分、时域、窗口和评价协议。",
        "engineering_mechanism": "所有候选模型使用同一冻结数据、切分、时域、窗口和评价协议。",
        "inference": "依据validation主指标选择模型，locked test只检查已选模型的泛化表现。",
        "expected_observation": "依据validation主指标选择模型，locked test只检查已选模型的泛化表现。",
        "verification_intent": "执行注册候选模型的同协议公平比较。",
        "falsification_condition": "候选模型无法在冻结validation主指标上形成可裁决结果。",
        "confirmation_criteria": confirmation,
        "falsification_criteria": falsification,
        "source_observation_ids": list(dict.fromkeys(
            str(value) for item in originals
            for value in item.get("source_observation_ids", [])
        )),
        "source_experiment_ids": list(dict.fromkeys(
            str(value) for item in originals
            for value in item.get("source_experiment_ids", [])
        )),
        "evidence_ids": list(dict.fromkeys(
            str(value) for item in originals
            for value in item.get("evidence_ids", [])
        )),
        "trigger_types": list(dict.fromkeys(
            str(value) for item in originals
            for value in item.get("trigger_types", [])
        )),
        "scientific_design": design.model_dump(mode="json"),
        "scientific_design_sha256": frozen_design_sha256(design),
        "experiment_intent": {"task_type": "model_comparison", "candidate_models": models},
        "compilation_status": "ADAPTED",
        "workflow_status": "COMPILED",
    })
    record = CompiledHypothesis(
        original_hypothesis_id=hypothesis_id,
        executable_hypothesis_id=hypothesis_id,
        original_claim=str(original.get("hypothesis") or ""),
        compiled_hypothesis=candidate,
        supported_operations=required_operations,
        required_operations=required_operations,
        adaptation_reason="model_comparison_problem_frozen_deterministically",
        current_executable=bool(len(models) >= 2),
        experiment_coverage=1.0 if len(models) >= 2 else 0.0,
        experiment_intent=candidate["experiment_intent"],
    )
    candidate["hypothesis_compilation"] = record.model_dump(
        mode="json", exclude={"compiled_hypothesis"}
    )
    return candidate, record.model_copy(update={"compiled_hypothesis": candidate})


class ClaimClassification(BaseModel):
    claim_type: Literal[
        "numeric_claim",
        "causal_claim",
        "statistical_claim",
        "unsupported_operation_claim",
    ]
    text: str = Field(min_length=1)


class CompiledHypothesis(BaseModel):
    original_hypothesis_id: str = Field(min_length=1)
    executable_hypothesis_id: str = Field(min_length=1)
    original_claim: str = Field(min_length=1)
    compiled_hypothesis: dict[str, Any]
    removed_claims: list[ClaimClassification] = Field(default_factory=list)
    unsupported_claims: list[ClaimClassification] = Field(default_factory=list)
    supported_operations: list[str] = Field(default_factory=list)
    required_operations: list[str] = Field(default_factory=list)
    adaptation_reason: str = Field(min_length=1)
    current_executable: bool
    experiment_coverage: float = Field(ge=0.0, le=1.0)
    experiment_intent: dict[str, Any] = Field(default_factory=dict)


def _scientific_text(hypothesis: dict[str, Any]) -> str:
    return " ".join(
        str(hypothesis.get(key, ""))
        for key in (
            "title", "hypothesis", "mechanism", "inference",
            "verification_intent", "falsification_condition",
        )
    )


def classify_claims(
    hypothesis: dict[str, Any],
    missing_capabilities: list[str] | None = None,
) -> list[ClaimClassification]:
    """Classify risky claims deterministically; no LLM judgment is used."""

    text = _scientific_text(hypothesis)
    claims: list[ClaimClassification] = []
    numeric = sorted(set(re.findall(
        r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:%|％)?", text
    )))
    for value in numeric:
        claims.append(ClaimClassification(claim_type="numeric_claim", text=value))
    causal_terms = [
        term for term in ("导致", "造成", "影响机制", "因果", "causes", "causal")
        if term.casefold() in text.casefold()
    ]
    for term in dict.fromkeys(causal_terms):
        claims.append(ClaimClassification(claim_type="causal_claim", text=term))
    statistical_terms = [
        term for term in (
            "显著", "统计显著", "p-value", "p value", "confidence interval", "置信区间"
        )
        if term.casefold() in text.casefold()
    ]
    for term in dict.fromkeys(statistical_terms):
        claims.append(ClaimClassification(claim_type="statistical_claim", text=term))
    for capability in sorted(set(missing_capabilities or [])):
        if capability.startswith("operation:"):
            claims.append(ClaimClassification(
                claim_type="unsupported_operation_claim",
                text=capability.split(":", 1)[1],
            ))
    return claims


def _next_id(original_id: str, used_ids: set[str]) -> str:
    candidate = f"{original_id}-A"
    index = 2
    while candidate in used_ids:
        candidate = f"{original_id}-A{index}"
        index += 1
    used_ids.add(candidate)
    return candidate


def _regime_variant(
    original: dict[str, Any],
    compiled_id: str,
    problem: dict[str, Any],
    removed: list[ClaimClassification],
    required_operations: list[str],
) -> tuple[dict[str, Any], str]:
    original_id = str(original.get("hypothesis_id") or original.get("id"))
    target = str(problem.get("target_variable") or "目标变量")
    condition = str(problem.get("operating_condition") or "预声明运行范围")
    statement = (
        f"在{condition}下，ramp_up工况的{target}预测MAE高于"
        "ramp_down工况。"
    )
    reason = (
        "将复杂假设编译为现有regime_stratified_evaluation可检验的"
        "可观测代理；数字幅度、因果机制、统计显著性及未注册操作均不进入结论。"
    )
    record = {
        "original_hypothesis_id": original_id,
        "adapted_hypothesis_id": compiled_id,
        "removed_claims": [
            f"{item.claim_type}:{item.text}" for item in removed
        ],
        "supported_operations": [REGIME_OPERATION],
        "reason": reason,
    }
    variant = deepcopy(original)
    variant.update({
        "id": compiled_id,
        "hypothesis_id": compiled_id,
        "title": "工况分层预测误差可执行变体",
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "mechanism": "能力边界编译变体仅检验预声明工况的可观测预测误差。",
        "engineering_mechanism": "能力边界编译变体仅检验预声明工况的可观测预测误差。",
        "inference": statement,
        "expected_observation": statement,
        "verification_intent": (
            "使用regime_stratified_evaluation分别计算ramp_up、"
            "ramp_down与steady工况的MAE。"
        ),
        "falsification_condition": (
            "所有已执行模型的ramp_up MAE均未高于ramp_down MAE。"
        ),
        "confirmation_criteria": [
            "all_models_regime_metric_greater:ramp_up|ramp_down|MAE"
        ],
        "falsification_criteria": [
            "all_models_regime_metric_not_greater:ramp_up|ramp_down|MAE"
        ],
        "variables": [target],
        "key_variables": [target],
        "original_claim": str(original.get("hypothesis") or ""),
        "original_hypothesis": deepcopy(original),
        "original_hypothesis_id": original_id,
        "adaptation_reason": reason,
        "removed_claims": [item.model_dump(mode="json") for item in removed],
        "unsupported_claims": [item.model_dump(mode="json") for item in removed],
        "unsupported_extensions": [
            item.text for item in removed
            if item.claim_type == "unsupported_operation_claim"
        ],
        "feasibility_adaptation": record,
        "compilation_status": "ADAPTED",
        "workflow_status": "COMPILED",
    })
    frozen = FrozenHypothesisDesign(
        experiment_type=REGIME_OPERATION,
        required_operations=[REGIME_OPERATION],
        required_targets=[target],
        required_metrics=["MAE"],
        prediction_horizon_steps=problem.get("required_horizon_steps"),
        control={"regime": "ramp_down"},
        treatment={"regime": "ramp_up"},
        confirmation_criteria=list(variant["confirmation_criteria"]),
        falsification_criteria=list(variant["falsification_criteria"]),
    )
    variant["scientific_design"] = frozen.model_dump(mode="json")
    variant["scientific_design_sha256"] = frozen_design_sha256(frozen)
    variant["compiler_required_operations"] = list(required_operations)
    return variant, reason


def _parameter_optimization_variant(
    original: dict[str, Any],
    compiled_id: str,
    problem: dict[str, Any],
    removed: list[ClaimClassification],
    capability_snapshot: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    original_id = str(original.get("hypothesis_id") or original.get("id"))
    variable = str(problem.get("optimization_variable") or "")
    candidates = list(problem.get("candidate_values") or [])
    intent = {
        "task_type": "parameter_optimization",
        "variable": variable,
        "candidates": candidates,
    }
    reason = (
        "Preserved the requested parameter-optimization semantics and removed "
        "only claims that exceed the registered experimental capability."
    )
    variant = deepcopy(original)
    original_statement = str(
        original.get("hypothesis") or original.get("hypothesis_statement") or ""
    )
    target = str(problem.get("target_variable") or "target_variable")
    condition = str(problem.get("operating_condition") or "declared operating range")
    statement = (
        f"在{condition}下，预声明的{variable}候选配置对{target}的"
        "验证集预测性能存在可比较差异。"
    )
    executable_mechanism = (
        f"{variable}改变模型可利用的历史输入范围；本执行变体只比较"
        "预声明候选配置的预测指标，不验证原假设中的因果、显著性或数值幅度声明。"
    )
    expected = "按预声明验证指标比较候选配置并识别表现最优的候选。"
    falsification = "所有候选配置在预声明验证指标上均无法形成可判定差异。"
    variant.update({
        "id": compiled_id,
        "hypothesis_id": compiled_id,
        "title": "参数候选预测性能可执行比较",
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "mechanism": executable_mechanism,
        "engineering_mechanism": executable_mechanism,
        "inference": expected,
        "expected_observation": expected,
        "verification_intent": expected,
        "falsification_condition": falsification,
        "evidence_gap": "需要完成全部预声明候选配置的同协议实验。",
        "variables": [variable, target],
        "key_variables": [variable, target],
        "original_claim": original_statement,
        "original_hypothesis": deepcopy(original),
        "original_hypothesis_id": original_id,
        "adaptation_reason": reason,
        "removed_claims": [item.model_dump(mode="json") for item in removed],
        "unsupported_claims": [item.model_dump(mode="json") for item in removed],
        "experiment_intent": intent,
        "compilation_status": "ADAPTED" if compiled_id != original_id else "UNCHANGED",
        "workflow_status": "COMPILED",
    })
    canonical_metrics = [
        metric for metric in ("MAE", "RMSE", "R2", "MBE")
        if any(metric in str(value).upper() for value in problem.get("metrics", []))
    ]
    available_metrics = set(capability_snapshot.get("available_metrics", []))
    metrics = canonical_metrics or [
        metric for metric in ("MAE", "RMSE", "R2")
        if not available_metrics or metric in available_metrics
    ]
    frozen = FrozenHypothesisDesign(
        experiment_type=PARAMETER_OPERATION,
        required_operations=[PARAMETER_OPERATION],
        required_targets=[target],
        required_metrics=metrics,
        prediction_horizon_steps=problem.get("required_horizon_steps"),
        control={},
        treatment={"parameter": variable, "candidate_values": candidates},
        confirmation_criteria=["all_candidates_worse_than_reference_on:MAE"],
        falsification_criteria=["any_candidate_better_than_reference_on:MAE"],
    )
    variant["scientific_design"] = frozen.model_dump(mode="json")
    variant["scientific_design_sha256"] = frozen_design_sha256(frozen)
    return variant, reason, intent


def compile_hypotheses(
    hypotheses: list[dict[str, Any]],
    problem: dict[str, Any],
    capability_snapshot: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[CompiledHypothesis]]:
    """Compile generated hypotheses into the currently executable envelope."""

    if hypotheses and _is_model_comparison_problem(problem):
        candidate, record = _model_comparison_candidate(
            hypotheses, problem, capability_snapshot
        )
        return [candidate], [record]

    supported_registry_operations = set(
        capability_snapshot.get("supported_experiment_operations", [])
    )
    used_ids = {
        str(item.get("hypothesis_id") or item.get("id") or "")
        for item in hypotheses
    }
    candidates: list[dict[str, Any]] = []
    records: list[CompiledHypothesis] = []
    for original in hypotheses:
        original_id = str(
            original.get("hypothesis_id") or original.get("id") or ""
        ).strip()
        if not original_id:
            continue
        admission = evaluate_candidate(original, problem, capability_snapshot)
        required = list(admission["required_operations"])
        supported = sorted(set(required) & supported_registry_operations)
        classifications = classify_claims(
            original, list(admission["missing_capabilities"])
        )
        parameter_task = (
            problem.get("research_task_type") == "parameter_optimization"
            and problem.get("optimization_variable")
            and problem.get("candidate_values")
        )
        if parameter_task:
            if PARAMETER_OPERATION not in supported_registry_operations:
                rejected = deepcopy(original)
                rejected["compilation_status"] = "UNSUPPORTED"
                reason = "parameter_optimization_operation_not_supported"
                record = CompiledHypothesis(
                    original_hypothesis_id=original_id,
                    executable_hypothesis_id=original_id,
                    original_claim=str(original.get("hypothesis") or ""),
                    compiled_hypothesis=rejected,
                    unsupported_claims=classifications,
                    required_operations=[PARAMETER_OPERATION],
                    adaptation_reason=reason,
                    current_executable=False,
                    experiment_coverage=0.0,
                    experiment_intent={
                        "task_type": "parameter_optimization",
                        "variable": problem.get("optimization_variable"),
                        "candidates": list(problem.get("candidate_values") or []),
                    },
                )
                rejected["hypothesis_compilation"] = record.model_dump(
                    mode="json", exclude={"compiled_hypothesis"}
                )
                candidates.append(rejected)
                records.append(record.model_copy(update={"compiled_hypothesis": rejected}))
                continue
            compiled_id = (
                original_id if admission["current_executable"] and not classifications
                else _next_id(original_id, used_ids)
            )
            variant, reason, intent = _parameter_optimization_variant(
                original, compiled_id, problem, classifications, capability_snapshot
            )
            record = CompiledHypothesis(
                original_hypothesis_id=original_id,
                executable_hypothesis_id=compiled_id,
                original_claim=str(original.get("hypothesis") or ""),
                compiled_hypothesis=variant,
                removed_claims=classifications,
                unsupported_claims=classifications,
                supported_operations=[PARAMETER_OPERATION],
                required_operations=[PARAMETER_OPERATION],
                adaptation_reason=reason,
                current_executable=True,
                experiment_coverage=1.0,
                experiment_intent=intent,
            )
            variant["hypothesis_compilation"] = record.model_dump(
                mode="json", exclude={"compiled_hypothesis"}
            )
            candidates.append(variant)
            records.append(record.model_copy(update={"compiled_hypothesis": variant}))
            continue
        unsafe = bool(classifications)
        if admission["current_executable"] and not unsafe:
            unchanged = deepcopy(original)
            unchanged["compilation_status"] = "UNCHANGED"
            unchanged["workflow_status"] = "COMPILED"
            record = CompiledHypothesis(
                original_hypothesis_id=original_id,
                executable_hypothesis_id=original_id,
                original_claim=str(original.get("hypothesis") or ""),
                compiled_hypothesis=unchanged,
                supported_operations=required,
                required_operations=required,
                adaptation_reason="hypothesis_fully_supported_unchanged",
                current_executable=True,
                experiment_coverage=1.0,
            )
            unchanged["hypothesis_compilation"] = record.model_dump(
                mode="json", exclude={"compiled_hypothesis"}
            )
            candidates.append(unchanged)
            records.append(record.model_copy(update={"compiled_hypothesis": unchanged}))
            continue

        if REGIME_OPERATION in supported:
            compiled_id = _next_id(original_id, used_ids)
            variant, reason = _regime_variant(
                original, compiled_id, problem, classifications, required
            )
            coverage = len(supported) / max(len(required), 1)
            record = CompiledHypothesis(
                original_hypothesis_id=original_id,
                executable_hypothesis_id=compiled_id,
                original_claim=str(original.get("hypothesis") or ""),
                compiled_hypothesis=variant,
                removed_claims=classifications,
                unsupported_claims=classifications,
                supported_operations=[REGIME_OPERATION],
                required_operations=required,
                adaptation_reason=reason,
                current_executable=True,
                experiment_coverage=coverage,
            )
            variant["hypothesis_compilation"] = record.model_dump(
                mode="json", exclude={"compiled_hypothesis"}
            )
            candidates.append(variant)
            records.append(record.model_copy(update={"compiled_hypothesis": variant}))
            continue

        rejected = deepcopy(original)
        rejected["compilation_status"] = "UNSUPPORTED"
        reason = "no_supported_experiment_operation_for_claim"
        record = CompiledHypothesis(
            original_hypothesis_id=original_id,
            executable_hypothesis_id=original_id,
            original_claim=str(original.get("hypothesis") or ""),
            compiled_hypothesis=rejected,
            removed_claims=[],
            unsupported_claims=classifications,
            supported_operations=supported,
            required_operations=required,
            adaptation_reason=reason,
            current_executable=False,
            experiment_coverage=(len(supported) / max(len(required), 1)),
        )
        rejected["hypothesis_compilation"] = record.model_dump(
            mode="json", exclude={"compiled_hypothesis"}
        )
        candidates.append(rejected)
        records.append(record.model_copy(update={"compiled_hypothesis": rejected}))
    return candidates, records
