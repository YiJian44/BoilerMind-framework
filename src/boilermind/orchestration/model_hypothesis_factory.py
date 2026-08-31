"""model_hypothesis_factory.py — 数据属性画像驱动的逐模型候选假设（确定性骨架）。

画像 → 每个候选模型一条假设 H_M（"模型 X 软测V误差最小" + 属性机理理由），
保证全覆盖、结构稳定。之后可选由 Qwen 检索/润色补充机理与证据（保留 LLM 痕迹），
但骨架是确定性的，不依赖 Qwen 自由生成。

生成的 H_M 会走现有 deterministic gate / compile / ranking。
"""
from __future__ import annotations

from typing import Any

from boilermind.core.contracts import DataProfile, ModelSelectionPlan
from boilermind.planning.experiment_requirement_parser import (
    FrozenHypothesisDesign,
    frozen_design_sha256,
)


def enrich_hypotheses_with_qwen(
    hypotheses: list[dict[str, Any]],
    *,
    profile: DataProfile | None = None,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """可选：Qwen 检索/润色补充每条假设的机理与证据（保留 LLM 痕迹）。

    确定性骨架不变（statement/证伪/grounding 保留），只由 Qwen 扩充
    engineering_mechanism 的科学叙述并引用检索到的证据；失败时静默保留原假设。
    """
    if not enabled or not hypotheses:
        return hypotheses
    try:
        from boilermind.core.llm_client import LLMClient

        client = LLMClient()
        attribute_summary = "；".join(
            f"{p.label}:{p.verdict}"
            for p in (profile.properties.values() if profile else [])
        )
    except Exception:
        return hypotheses

    enriched: list[dict[str, Any]] = []
    for h in hypotheses:
        family = h.get("model_family") or ""
        prompt = (
            "你是 BoilerMind 锅炉软测量假设机理补充器。给定候选模型与数据属性判定，"
            "把该模型假设的 engineering_mechanism 扩充成一段有科学依据的叙述（3-5句），"
            "说明该模型为何适配这些数据属性来软测蒸汽体积量V。不得添加具体数值/百分比，"
            "不得改变'模型X软测V误差最小'这一核心主张，不得编造evidence_id。\n\n"
            f"候选模型：{family}\n"
            f"数据属性判定：{attribute_summary}\n"
            f"原机理：{h.get('engineering_mechanism', '')}"
        )
        try:
            rationale = str(client.generate(prompt)).strip()
            if len(rationale) > 20:
                h["engineering_mechanism"] = (
                    f"{rationale}（Qwen 补充；原确定性机理：{h.get('engineering_mechanism', '')}）"
                )
                h["mechanism"] = h["engineering_mechanism"]
        except Exception:
            pass
        enriched.append(h)
    return enriched

# 模型家族 → 数据属性机理关键词（与画像判定联动）
FAMILY_AFFINITY: dict[str, str] = {
    "ridge": "线性主效应 + 强共线 → 岭回归正则化稳定",
    "bayesianridge": "线性主效应 + 强共线 → 贝叶斯岭正则化稳定且概率化",
    "elasticnet": "线性主效应 + 稀疏 → 弹性网折中 L1/L2",
    "pls": "强共线 + 内蕴维低 → 潜变量投影提取",
    "svr": "中度非线性 → 核方法拟合非线性边界",
    "rf": "非线性 + 交互 → 随机森林集成",
    "hgb": "非线性 + 重尾 → 梯度提升树",
    "mlp": "非线性 → 多层感知机",
    "knn": "非参数局部结构 → 最近邻",
    "lstm": "强时序 + 非平稳 → 循环记忆",
    "gru": "强时序 + 非平稳 → 门控循环",
    "dlinear": "时序 + 线性 → 可分解线性时序",
    "transformer": "长程时序依赖 → 自注意力",
}

PROPERTY_LABELS = {
    "temporal": "时序性", "nonlinearity": "非线性", "non_gaussian": "非高斯",
    "singular_outlier": "奇异值/共线", "sparsity": "稀疏化", "dimensionality": "降维",
}


def _attribute_evidence(profile: DataProfile) -> list[str]:
    """从画像判定提炼"数据属性→模型族"的机理证据文本。"""
    lines = []
    for key, prop in profile.properties.items():
        label = PROPERTY_LABELS.get(key, key)
        verdict = prop.verdict
        cand = prop.candidate_families or prop.points_to
        lines.append(f"数据属性[{label}]判定：{verdict}；指向模型族 {cand}")
    return lines


def build_model_hypotheses(
    profile: DataProfile,
    plan: ModelSelectionPlan,
    *,
    problem_id: str | None = None,
    valid_evidence_ids: list[str] | None = None,
    valid_observation_ids: list[str] | None = None,
    valid_experiment_ids: list[str] | None = None,
    horizon_label: str = "当前时刻",
) -> list[dict[str, Any]]:
    """确定性逐模型候选假设 H_M（每个候选模型一条）。"""
    attribute_lines = _attribute_evidence(profile)
    attribute_text = "；".join(attribute_lines)
    evidence_ids = list(valid_evidence_ids or [])[:6]
    observation_ids = list(valid_observation_ids or [])[:6]
    experiment_ids = list(valid_experiment_ids or [])[:6]
    hypotheses = []
    for index, family in enumerate(plan.to_run_families, start=1):
        affinity = FAMILY_AFFINITY.get(family, "与数据属性匹配")
        mechanism = (
            f"数据属性画像判定该数据具有以下特性：{attribute_text}。"
            f"{family} 的机理优势：{affinity}，因此假设其在软测蒸汽体积量 V "
            f"（{horizon_label}）上误差最小。"
        )
        falsification = (
            f"若 {family} 的 validation MAE 不是所有候选中最小的，"
            f"或未优于无-V 均值基线，则本假设被证伪"
        )
        h = {
            "hypothesis_id": f"H{index:03d}",
            "id": f"H{index:03d}",
            "title": f"模型 {family} 软测蒸汽体积量 V 误差最小",
            "hypothesis": f"模型 {family} 在软测蒸汽体积量 V 上误差最小",
            "hypothesis_statement": f"模型 {family} 在软测蒸汽体积量 V 上误差最小",
            "mechanism": mechanism,
            "engineering_mechanism": mechanism,
            "inference": f"该模型的 validation MAE 应低于其他候选模型",
            "expected_observation": f"该模型的 validation MAE 应低于其他候选模型",
            "verification_intent": (
                f"在相同数据/切分/种子下运行模型 {family}，"
                f"比较 validation MAE 与其余候选及无-V 基线"
            ),
            "falsification_condition": falsification,
            "evidence_gap": "需要该模型在真实软测数据上的 validation/locked_test MAE",
            "key_variables": ["steam_volumetric_flow"],
            "variables": ["steam_volumetric_flow"],
            "applicability_conditions": ["软测蒸汽体积量 V，当前时刻"],
            "assumptions": ["31 特征→V 软测口径；均值基线代表无-V 基准"],
            "evidence_needed": ["该模型的软测 MAE"],
            "evidence_ids": list(evidence_ids),
            "source_observation_ids": list(observation_ids),
            "source_experiment_ids": list(experiment_ids),
            "confirmation_criteria": [
                "candidate_validation_mae_minimum",
                "candidate_locked_test_generalization",
            ],
            "falsification_criteria": [
                "candidate_validation_mae_not_minimum",
                "candidate_locked_test_not_generalized",
            ],
            "trigger_types": ["HUMAN_PROPOSAL"],
            "generation_source": "data_profile_deterministic",
            "problem_id": problem_id,
            "target_variable": "steam_volumetric_flow",
            "model_family": family,
            "required_models": [family],
            "current_executable": True,
            "verification_mapping": {
                "executable_now": True,
                "recommended_action": "EXECUTE_NOW",
                "supported_operations": [
                    "model_comparison", "chronological_validation",
                    "locked_test_evaluation",
                ],
                "missing_capabilities": [],
                "missing_operations": [],
                "required_user_inputs": [],
            },
            "data_attribute_prior": float(
                plan.property_scores.get(family, 0)
            ) / max(
                max(plan.property_scores.values(), default=1), 1
            ),
        }
        # 冻结科学设计：让 PlanningSkill 走 requirements_from_frozen_design，
        # 不再对假设文本做 experiment_type 分类（避免 unsupported_scientific_design）。
        frozen = FrozenHypothesisDesign(
            experiment_type="model_comparison",
            required_operations=[
                "model_comparison", "chronological_validation",
                "locked_test_evaluation",
            ],
            required_models=[family],
            required_model_roles={family: "candidate"},
            required_targets=["steam_volumetric_flow"],
            required_metrics=["MAE", "RMSE", "R2", "MBE"],
            prediction_horizon_steps=0,
            confirmation_criteria=[
                "candidate_validation_mae_minimum",
                "candidate_locked_test_generalization",
            ],
            falsification_criteria=[
                "candidate_validation_mae_not_minimum",
                "candidate_locked_test_not_generalized",
            ],
            hard_constraints=[],
            requires_feature_intervention=False,
        )
        h["scientific_design"] = frozen.model_dump(mode="json")
        h["scientific_design_sha256"] = frozen_design_sha256(frozen)
        hypotheses.append(h)
    return hypotheses
