from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class ClarificationItem(BaseModel):
    field: str = Field(min_length=1)
    question: str = Field(min_length=1)
    choices: list[str] = Field(default_factory=list)
    required: bool = True


class ProblemIntakeDecision(BaseModel):
    schema_version: str = "boilermind.problem_intake.v1"
    status: str
    problem_type: str
    currently_executable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extracted: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    clarification_items: list[ClarificationItem] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    next_action: str


_TYPE_RULES: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("sensor_fault_robustness", ("噪声", "毛刺", "尖峰", "漂移", "传感器冻结", "仪表冻结", "冻结不变", "数据缺失", "传感器缺失", "传感器故障", "仪表故障"), [
        "sensor_corruption_injection", "clean_corrupted_paired_comparison",
        "robustness_degradation_evaluation",
    ]),
    ("direction_rate_interaction", ("升负荷", "降负荷", "变化速率", "变负荷速率", "ramp_up", "ramp_down"), [
        "load_rate_computation", "direction_regime_assignment",
        "rate_stratified_evaluation", "direction_rate_interaction_evaluation",
    ]),
    ("regime_comparison", ("工况分层", "稳态", "方向切换", "不同工况", "低负荷区", "高负荷区"), [
        "regime_assignment", "regime_stratified_evaluation",
    ]),
    ("multi_seed_stability", ("多seed", "多 seed", "随机种子", "排名翻转", "seed 7"), [
        "multi_seed_execution", "seed_aggregate_evaluation",
    ]),
    ("time_block_generalization", ("时间块", "跨时间", "不同月份", "跨时段", "时间泛化"), [
        "time_block_split", "cross_time_block_evaluation",
    ]),
    ("resource_benchmark", ("算力", "cpu", "gpu", "推理延迟", "运行时间", "内存", "现场跑得动", "计算成本"), [
        "resource_benchmark", "inference_latency_evaluation",
    ]),
    ("uncertainty_evaluation", ("不确定性", "置信度", "预测区间", "过度自信"), [
        "uncertainty_estimation", "interval_calibration_evaluation",
    ]),
    ("causal_mechanism", ("导致", "因果", "根本原因", "为什么"), [
        "causal_design", "confounder_control", "causal_effect_estimation",
    ]),
    ("control_optimization", ("怎么调", "如何调", "控制策略", "优化运行", "最大化", "约束优化"), [
        "control_policy_evaluation", "constraint_optimization",
    ]),
    ("feature_selection", ("变量组合", "传感器组合", "删除变量", "冗余变量", "特征选择", "消融"), [
        "feature_intervention", "feature_subset_evaluation",
    ]),
]


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _target(text: str) -> str | None:
    if _contains_any(text, ("蒸汽体积流量", "主蒸汽体积流量", "steam volumetric flow")):
        return "steam_volumetric_flow"
    if _contains_any(text, ("蒸汽质量流量", "主蒸汽质量流量", "steam mass flow")):
        return "main_steam_mass_flow"
    return None


def _horizon(text: str) -> int | None:
    match = re.search(r"(?<![a-z0-9_])h\s*(80|40|20|8|4)(?![0-9])", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(80|40|20|8|4)\s*步", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(20|10|5|2|1)\s*分钟", text)
    if match:
        return {"20": 80, "10": 40, "5": 20, "2": 8, "1": 4}[match.group(1)]
    return None


def _metrics(text: str) -> list[str]:
    aliases = (
        ("MAE", ("mae", "平均绝对误差")),
        ("RMSE", ("rmse", "均方根误差")),
        ("R2", ("r2", "决定系数")),
        ("MBE", ("mbe", "平均偏差")),
    )
    return [name for name, terms in aliases if _contains_any(text, terms)]


def _resource_limit(text: str) -> str | None:
    cpu = re.search(r"(\d+)\s*核\s*cpu", text)
    if cpu:
        return f"{cpu.group(1)}核CPU"
    latency = re.search(r"(?:推理延迟|延迟)[^\d]{0,8}(\d+(?:\.\d+)?)\s*(ms|毫秒|s|秒)", text)
    if latency:
        return f"{latency.group(1)}{latency.group(2)}"
    memory = re.search(r"(?:内存)[^\d]{0,8}(\d+(?:\.\d+)?)\s*(gb|mb)", text)
    if memory:
        return f"{memory.group(1)}{memory.group(2).upper()}"
    return None


def _models(text: str) -> tuple[list[str], list[str]]:
    rules = (
        ("persistence", r"(?<![a-z0-9_])persistence(?![a-z0-9_])|持久性(?:基线)?"),
        ("bayesianridge", r"(?<![a-z0-9_])bayesianridge(?![a-z0-9_])|贝叶斯岭"),
        ("ridge", r"(?<![a-z0-9_])ridge(?![a-z0-9_])|(?<!贝叶斯)岭回归"),
        ("hgb", r"(?<![a-z0-9_])hgb(?![a-z0-9_])"),
        ("rf", r"(?<![a-z0-9_])rf(?![a-z0-9_])|随机森林"),
        ("lstm", r"(?<![a-z0-9_])lstm(?![a-z0-9_])"),
        ("transformer", r"(?<![a-z0-9_])transformer(?![a-z0-9_])"),
    )
    found = [name for name, pattern in rules if re.search(pattern, text)]
    return [item for item in found if item != "persistence"], [
        item for item in found if item == "persistence"
    ]


def _clarification(field: str, problem_type: str) -> ClarificationItem:
    common = {
        "target_definition": ClarificationItem(
            field=field, question="你说的蒸汽量具体指什么？",
            choices=["蒸汽体积流量", "蒸汽质量流量"],
        ),
        "prediction_horizon": ClarificationItem(
            field=field, question="要估计当前值还是预测未来值？若预测未来，请选择范围。",
            choices=["当前软测量", "h40（10分钟）", "h80（20分钟）"],
        ),
        "objective": ClarificationItem(
            field=field, question="这次最主要想解决什么？",
            choices=["预测准确性", "不同工况稳定性", "仪表故障鲁棒性", "计算与部署成本", "解释影响机制"],
        ),
        "metrics": ClarificationItem(
            field=field, question="用哪些指标判断结果？",
            choices=["MAE+RMSE", "MAE+RMSE+MBE", "R2", "由系统建议后确认"],
        ),
        "candidate_models": ClarificationItem(
            field=field, question="比较哪些候选方法？",
            choices=["Ridge+BayesianRidge+HGB", "全部当前可执行模型", "我自行填写"],
        ),
        "reference_model": ClarificationItem(
            field=field, question="以什么作为参考？",
            choices=["Persistence", "当前生产模型", "指定模型"],
        ),
    }
    if field in common:
        return common[field]
    specialized = {
        "corruption_type": ("需要模拟哪类仪表问题？", ["噪声", "毛刺", "漂移", "冻结", "缺失"]),
        "corruption_levels": ("需要比较哪些故障强度？", ["轻/中/重三级", "按实际仪表规格", "自行填写"]),
        "regimes": ("需要比较哪些工况？", ["steady/ramp_up/ramp_down/direction_change", "低/中/高负荷", "自行填写"]),
        "resource_limit": ("现场计算约束是什么？", ["8核CPU", "指定最大推理延迟", "指定内存上限", "自行填写"]),
        "causal_exposure": ("希望检验哪个因素对哪个结果的影响？", ["自行填写因素与结果"]),
        "control_constraints": ("控制目标和安全约束分别是什么？", ["自行填写目标、可调变量和边界"]),
    }
    question, choices = specialized.get(
        field, (f"请补充 {field}。", ["自行填写"])
    )
    return ClarificationItem(field=field, question=question, choices=choices)


def analyze_problem_intake(question: str) -> ProblemIntakeDecision:
    text = question.strip().casefold()
    if not text:
        raise ValueError("research_question_required")
    candidates, references = _models(text)
    explicit_comparison = (
        bool(candidates or references)
        or _contains_any(text, ("比较", "对比", "优于", "劣于", "compare"))
    )
    objective = None
    if _contains_any(text, ("准确", "误差", "精度", "预测效果")):
        objective = "prediction_accuracy"
    problem_type = (
        "model_comparison" if explicit_comparison
        else "general_soft_sensor_research"
    )
    capabilities = (
        ["model_comparison", "locked_test_evaluation"]
        if explicit_comparison else []
    )
    for candidate, keywords, required in _TYPE_RULES:
        if _contains_any(text, keywords):
            problem_type, capabilities = candidate, required
            break
    if (
        problem_type == "general_soft_sensor_research"
        and objective == "prediction_accuracy"
    ):
        problem_type = "model_comparison"
        capabilities = ["model_comparison", "locked_test_evaluation"]
    target = _target(text)
    horizon = _horizon(text)
    metrics = _metrics(text)
    resource_limit = _resource_limit(text)
    extracted = {
        "target_variable": target,
        "prediction_horizon_steps": horizon,
        "metrics": metrics,
        "candidate_models": candidates,
        "reference_models": references,
        "objective": objective,
        "resource_limit": resource_limit,
    }
    # Hypothesis formation only needs a recognisable engineering object and
    # phenomenon.  Metrics, horizon, models and detailed intervention levels
    # are experiment-planning inputs and must not block scientific ideation.
    missing: list[str] = []
    if target is None and problem_type == "general_soft_sensor_research":
        missing.append("target_definition")
    if missing:
        return ProblemIntakeDecision(
            status="NEEDS_CLARIFICATION",
            problem_type=problem_type,
            currently_executable=False,
            confidence=0.95,
            extracted=extracted,
            missing_fields=list(dict.fromkeys(missing)),
            clarification_items=[
                _clarification(field, problem_type)
                for field in dict.fromkeys(missing)
            ],
            required_capabilities=capabilities,
            next_action="ASK_USER_AND_RESUBMIT",
        )
    return ProblemIntakeDecision(
        status="READY_FOR_HYPOTHESIS",
        problem_type=problem_type,
        # This flag describes the intake stage only.  Actual executability is
        # assessed independently for every generated hypothesis.
        currently_executable=True,
        confidence=0.95,
        extracted=extracted,
        required_capabilities=capabilities,
        next_action="CONTINUE_PIPELINE",
    )
