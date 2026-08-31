"""control_hypothesis_factory.py — 控制调参建议假设工厂（确定性，一条含调参范围的假设）。

生成一条假设："在压力≤limit下，将给煤/给水/送风/汽包压力调整至指定范围，
可使软测蒸汽体积量V上升 target_rise"。HGB 按范围软测V验证，成功后推给 Unity。
"""
from __future__ import annotations

from typing import Any

VAR_NAMES = ["给煤", "给水", "送风", "汽包压力"]
VAR_COLS = ["col5", "col14", "col17", "col2"]


def build_control_hypothesis(
    *,
    ranges: list[tuple[float, float]],        # 每变量建议范围 [min,max]（绝对量）
    current_values: list[float],              # 当前值
    predicted_rise: float,                    # HGB 按建议预测的 V 上升比例（如 0.15）
    target_rise: float = 0.15,
    pressure_limit: float = 23.0,
    target_variable: str = "steam_volumetric_flow",
    problem_id: str | None = None,
) -> dict[str, Any]:
    """生成一条带调参范围的联合调参假设（验证成功后供 Unity 推送）。"""
    range_text = "、".join(
        f"{name}→[{lo:.1f},{hi:.1f}]" for name, (lo, hi) in zip(VAR_NAMES, ranges)
    )
    mech = "联合增加燃烧输入（煤/水/风）协同提升蒸汽产量 -> V↑；汽包压力保持限制内"
    statement = (
        f"在压力≤{pressure_limit:.0f}MPa限制下，按范围调整（{range_text}），"
        f"可使软测蒸汽体积量V上升{target_rise*100:.0f}%"
    )
    return {
        "hypothesis_id": "H_CTRL",
        "id": "H_CTRL",
        "title": f"联合调参使软测V升{target_rise*100:.0f}%（压力≤{pressure_limit:.0f}MPa）",
        "hypothesis": statement,
        "hypothesis_statement": statement,
        "mechanism": mech,
        "mechanism_chain": mech,
        "engineering_mechanism": mech,
        "inference": f"按范围调整后V预测值应上升约{target_rise*100:.0f}%",
        "expected_observation": f"按范围调整后V预测值应上升约{target_rise*100:.0f}%",
        "verification_intent": (
            f"HGB 按建议范围软测V；若预测上升≥{target_rise*100:.0f}%则小模型验证成功，"
            f"推送Unity按范围调节并核对V"
        ),
        "falsification_condition": (
            f"若按建议范围调整后V预测未达目标（{target_rise*100:.0f}%±2%），则本假设被证伪"
        ),
        "evidence_gap": "需要Unity干预演示确认按范围调节后V确实上升",
        "key_variables": ["steam_volumetric_flow"] + VAR_COLS,
        "variables": ["steam_volumetric_flow"] + VAR_COLS,
        "applicability_conditions": [f"压力≤{pressure_limit:.0f}MPa", "稳态工况",
                                     "V在数据范围内（3.5-4.5）"],
        "assumptions": ["V=f(4变量)HGB模型有效；相关性非因果；列号为候选推断"],
        "evidence_needed": ["HGB按范围软测验证 + Unity干预演示"],
        "evidence_ids": [],
        "source_observation_ids": [],
        "source_experiment_ids": [],
        "trigger_types": ["HUMAN_PROPOSAL"],
        "generation_source": "control_hypothesis_deterministic",
        "problem_id": problem_id,
        "target_variable": target_variable,
        "control_ranges": {
            name: {"column": col, "current": cur, "range_min": lo, "range_max": hi}
            for name, col, cur, (lo, hi) in zip(VAR_NAMES, VAR_COLS, current_values, ranges)
        },
        "adjustment_ranges": [list(r) for r in ranges],  # 供 Unity 直接消费
        "pressure_limit_mpa": pressure_limit,
        "target_rise": target_rise,
        "predicted_rise": predicted_rise,
        "validated_on_small_model": bool(predicted_rise >= target_rise * 0.98),
    }
