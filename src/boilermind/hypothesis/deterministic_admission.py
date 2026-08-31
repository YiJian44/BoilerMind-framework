from __future__ import annotations

import re
from typing import Any

from boilermind.planning.experiment_requirement_parser import (
    FrozenHypothesisDesign,
    frozen_design_sha256,
    parse_hypothesis_requirements,
    requirements_from_frozen_design,
)


OBJECTIVE_ACCURACY = "accuracy"
OBJECTIVE_FAULT_ROBUSTNESS = "fault_robustness"
OBJECTIVE_DEPLOYMENT_EFFICIENCY = "deployment_efficiency"


def required_objective_dimensions(text: str) -> list[str]:
    lowered = str(text).casefold()
    dimensions: list[str] = []
    if any(token in lowered for token in (
        "准确", "精度", "误差", "估得准", "mae", "rmse", "r2",
    )):
        dimensions.append(OBJECTIVE_ACCURACY)
    if any(token in lowered for token in (
        "故障", "失效", "坏点", "漂移", "冻结", "缺失", "异常",
        "表出毛病", "仪表出毛病", "乱套",
        "鲁棒", "容错", "降级", "fallback", "fault", "robust",
    )):
        dimensions.append(OBJECTIVE_FAULT_ROBUSTNESS)
    if any(token in lowered for token in (
        "算力", "计算资源", "部署", "推理时间", "延迟", "内存",
        "模型大小", "跑得动", "不费太多", "cpu", "runtime", "latency", "memory", "throughput",
    )):
        dimensions.append(OBJECTIVE_DEPLOYMENT_EFFICIENCY)
    return list(dict.fromkeys(dimensions))


def _candidate_text(hypothesis: dict[str, Any]) -> str:
    return " ".join(str(hypothesis.get(key, "")) for key in (
        "title", "hypothesis", "mechanism", "verification_intent",
        "falsification_condition", "evidence_gap",
    )).casefold()


def objective_coverage(hypothesis: dict[str, Any]) -> dict[str, bool]:
    text = _candidate_text(hypothesis)
    return {
        OBJECTIVE_ACCURACY: any(token in text for token in (
            "mae", "rmse", "r2", "mbe", "准确", "精度", "误差", "估得准",
        )),
        OBJECTIVE_FAULT_ROBUSTNESS: (
            any(token in text for token in (
                "故障", "失效", "漂移", "冻结", "缺失", "尖峰", "异常",
                "鲁棒", "容错", "降级", "fallback", "fault", "robust",
            ))
            and any(token in text for token in (
                "故障注入", "缺失注入", "漂移注入", "冻结注入", "尖峰注入",
                "降级通道", "降级策略", "fallback", "fault injection",
                "degradation", "恢复时间", "灾难性误差",
            ))
        ),
        OBJECTIVE_DEPLOYMENT_EFFICIENCY: any(token in text for token in (
            "runtime", "latency", "推理时间", "推理延迟", "内存",
            "模型大小", "cpu", "吞吐", "资源占用", "计算资源消耗",
        )),
    }


def _canonical_metrics(values: list[Any]) -> list[str]:
    supported = ("MAE", "RMSE", "R2", "MBE")
    text = " ".join(str(value).upper() for value in values)
    return [metric for metric in supported if metric in text]


def _compiled_requirements(
    hypothesis: dict[str, Any],
) -> tuple[Any | None, str | None]:
    """Read a compiler-frozen design only when its provenance hash is valid."""
    compilation = hypothesis.get("hypothesis_compilation")
    intent = hypothesis.get("experiment_intent")
    frozen_payload = hypothesis.get("scientific_design")
    frozen_hash = str(hypothesis.get("scientific_design_sha256") or "")
    if not isinstance(compilation, dict) or not isinstance(intent, dict):
        return None, None
    if not frozen_payload or not frozen_hash:
        return None, "compiled_execution_intent_missing_frozen_design"
    try:
        frozen = FrozenHypothesisDesign.model_validate(frozen_payload)
    except Exception as exc:
        return None, f"compiled_execution_intent_invalid:{exc}"
    if frozen_design_sha256(frozen) != frozen_hash:
        return None, "compiled_execution_intent_sha256_mismatch"
    hypothesis_id = str(
        hypothesis.get("hypothesis_id") or hypothesis.get("id") or ""
    )
    return requirements_from_frozen_design(hypothesis_id, frozen), None


def extract_requirements(
    hypothesis: dict[str, Any],
    problem: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed, compiled_issue = _compiled_requirements(hypothesis)
    source = "compiled_execution_intent" if parsed is not None else "raw_hypothesis"
    if parsed is None and compiled_issue is None and isinstance(problem, dict):
        if (
            problem.get("research_task_type") == "parameter_optimization"
            and problem.get("optimization_variable")
            and problem.get("candidate_values")
        ):
            # Research-task fallback is deterministic and deliberately narrow.
            # The Compiler remains responsible for freezing the final design.
            raw = parse_hypothesis_requirements(hypothesis)
            return {
                "required_operations": ["model_comparison"],
                "required_models": sorted(raw.required_models),
                "required_metrics": (
                    _canonical_metrics(list(problem.get("metrics") or []))
                    or ["MAE", "RMSE", "R2"]
                ),
                "required_horizon_steps": problem.get("required_horizon_steps"),
                "required_variables": [],
                "execution_semantics_source": "research_task_type",
                "execution_intent_issue": None,
            }
    if parsed is None:
        parsed = parse_hypothesis_requirements(hypothesis)
    operations = set(parsed.required_operations)
    # 合并问题声明的实验操作（如数据属性画像注入的 model_comparison），
    # 避免纯模型比较假设因措辞未命中解析器关键词而被判 UNSUPPORTED。
    problem_ops = {
        str(op).strip()
        for op in (problem.get("required_operations") or [])
        if str(op).strip()
    } if isinstance(problem, dict) else set()
    operations.update(problem_ops)
    # 仅当问题显式声明了实验操作时，去掉解析器"不支持设计"兜底标记
    # （该标记是解析器无法分类时的 fail-closed 兜底，只有明确的问题类型才能覆盖）。
    if problem_ops and operations - {"unsupported_scientific_design"}:
        operations.discard("unsupported_scientific_design")
    metrics = set(parsed.required_metrics)
    if source == "raw_hypothesis":
        text = _candidate_text(hypothesis)
        if any(token in text for token in (
            "故障注入", "缺失注入", "漂移注入", "冻结注入", "尖峰注入",
            "fault injection",
        )):
            operations.add("sensor_fault_injection")
        if any(token in text for token in ("降级通道", "降级策略", "fallback")):
            operations.add("fallback_evaluation")
        if any(token in text for token in (
            "升负荷", "降负荷", "稳态", "变负荷方向", "工况分层",
            "ramp_up", "ramp_down", "regime",
        )):
            operations.add("regime_stratified_evaluation")
        if any(token in text for token in ("滞后", "延迟响应", "时滞", "lag")):
            operations.add("lag_analysis")
        if any(token in text for token in (
            "变量消融", "特征消融", "移除变量", "耦合贡献", "ablation",
        )):
            operations.add("feature_ablation")
        if any(token in text for token in ("runtime", "推理时间", "推理延迟", "latency")):
            operations.add("runtime_benchmark")
            metrics.add("runtime")
        if any(token in text for token in ("内存", "模型大小", "cpu", "资源占用", "throughput", "吞吐")):
            operations.add("resource_profile")
    return {
        "required_operations": sorted(operations),
        "required_models": sorted(parsed.required_models),
        "required_metrics": sorted(metrics),
        "required_horizon_steps": parsed.prediction_horizon_steps,
        "required_variables": sorted(parsed.required_variables),
        "execution_semantics_source": source,
        "execution_intent_issue": compiled_issue,
    }


def evaluate_candidate(
    hypothesis: dict[str, Any],
    problem: dict[str, Any],
    scientific_context: dict[str, Any],
) -> dict[str, Any]:
    requirements = extract_requirements(hypothesis, problem)
    supported_operations = set(scientific_context.get("supported_experiment_operations", []))
    enabled_models = set(scientific_context.get("enabled_experiment_models", []))
    reference = scientific_context.get("reference_model")
    if reference:
        enabled_models.add(str(reference))
    available_metrics = set(scientific_context.get("available_metrics", []))
    available_variables = set(scientific_context.get("available_variables", []))
    missing = [
        *[f"operation:{item}" for item in set(requirements["required_operations"]) - supported_operations],
        *[f"model:{item}" for item in set(requirements["required_models"]) - enabled_models],
        *[f"metric:{item}" for item in set(requirements["required_metrics"]) - available_metrics],
        *(
            [
                f"variable:{item}"
                for item in set(requirements["required_variables"]) - available_variables
            ]
            if available_variables else []
        ),
    ]
    if requirements.get("execution_intent_issue"):
        missing.append(str(requirements["execution_intent_issue"]))
    supported_horizons = {
        int(item)
        for item in scientific_context.get("supported_prediction_horizon_steps", [])
        if item is not None
    }
    required_horizon = requirements.get("required_horizon_steps")
    if (
        supported_horizons
        and required_horizon is not None
        and int(required_horizon) not in supported_horizons
    ):
        missing.append(
            "prediction_horizon_steps:"
            f"{int(required_horizon)};supported="
            + ",".join(map(str, sorted(supported_horizons)))
        )
    required_dimensions = list(problem.get("required_objective_dimensions", []))
    coverage = objective_coverage(hypothesis)
    missing_dimensions = [item for item in required_dimensions if not coverage.get(item, False)]
    duplicate = hypothesis.get("duplicate_check", {})
    duplicate_of = duplicate.get("duplicate_of") if isinstance(duplicate, dict) else None
    return {
        **requirements,
        "required_objective_dimensions": required_dimensions,
        "objective_coverage": coverage,
        "missing_objective_dimensions": missing_dimensions,
        "objective_coverage_ratio": (
            1.0 if not required_dimensions else
            round((len(required_dimensions) - len(missing_dimensions)) / len(required_dimensions), 4)
        ),
        "missing_capabilities": sorted(missing),
        "current_executable": not missing,
        "duplicate_of_completed_experiment": duplicate_of,
        "hard_rejection_reasons": ([f"duplicate_of_completed_experiment:{duplicate_of}"] if duplicate_of else []),
        "eligible_as_comprehensive_champion": not missing_dimensions and not missing and not duplicate_of,
    }
