"""profile_mapper.py — 数据属性 → 模型库 14 家族选型决策（确定性）。

读 DataProfile，按 6 项属性的 candidate_families 统计每个模型家族被指向次数
（property_scores），生成推荐集合（score>=2）+ 固定基线，映射到模型库具体 id。

输出 ModelSelectionPlan，作为"逐模型候选假设池"（H_M）与基线对照的依据。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from boilermind.core.contracts.data_profile import (
    DataProfile,
    ModelSelectionPlan,
)

# 深度族 / 树族 / 经典核心
DEEP = {"lstm", "gru", "dlinear", "transformer"}
TREE = {"rf", "hgb"}
CLASSIC_CORE = {"ridge", "bayesianridge"}

# 并列时固定优先序（ridge > bayesianridge > hgb > pls > lstm > gru > dlinear
# > transformer > rf > mlp > svr > knn > elasticnet）
_PRIORITY = {
    "ridge": 0, "bayesianridge": 1, "hgb": 2, "pls": 3,
    "lstm": 4, "gru": 5, "dlinear": 6, "transformer": 7,
    "rf": 8, "mlp": 9, "svr": 10, "knn": 11, "elasticnet": 12,
}

# 模型库注册表路径（权威口径）
_LIBRARY_PATH = (
    Path(__file__).resolve().parents[3]
    / "model_library"
    / "model_library.json"
)


def _library_ids(library_path: Path | None = None) -> set[str]:
    """读取 model_library.json 中的模型 id 集合（如 ridge_h40）。"""
    path = library_path or _LIBRARY_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    ids = set()
    for model in payload.get("models", []):
        mid = str(model.get("id") or "").strip()
        if mid:
            ids.add(mid)
    return ids


def profile_to_model_selection(
    profile: DataProfile,
    *,
    horizon_steps: int,
    library_path: Path | None = None,
) -> ModelSelectionPlan:
    """把数据属性画像转成模型库选型计划（确定性规则）。"""

    # 1) 每个家族被几项属性指向
    score: Counter[str] = Counter()
    rationale: list[str] = []
    for key, prop in profile.properties.items():
        for family in prop.candidate_families:
            score[family] += 1
        target = prop.candidate_families or prop.points_to
        rationale.append(
            f"{prop.label} 判定 '{prop.verdict}' → 指向 {target}"
        )

    # 2) 推荐 = 被 >=2 项属性指向；不足 2 个取 top2（按固定优先序）
    ordered = sorted(
        score,
        key=lambda f: (
            -score[f],
            _PRIORITY.get(f, 99),
            f,
        ),
    )
    recommended = [f for f in ordered if score[f] >= 2]
    if len(recommended) < 2:
        recommended = ordered[:2]

    # 3) 固定基线（保证"说得通"的对照）
    classic = sorted(CLASSIC_CORE)
    strong: list[str] = []
    if not (set(recommended) & DEEP):
        strong = ["lstm", "transformer"]
    to_run = list(dict.fromkeys(recommended + classic + strong))
    if len(to_run) < 5 and not (set(to_run) & TREE):
        to_run.append("hgb")

    # 4) 映射到模型库 id（如 ridge_h40），与库交叉校验
    to_run = to_run[:11]
    to_run_library_ids = [f"{f}_h{horizon_steps}" for f in to_run]
    valid = _library_ids(library_path)
    missing_library_ids = (
        [i for i in to_run_library_ids if i not in valid]
        if valid
        else []
    )

    return ModelSelectionPlan(
        selection_id=f"SEL-{uuid4().hex[:12]}",
        profile_id=profile.profile_id,
        horizon_steps=horizon_steps,
        recommended_families=sorted(recommended),
        baseline_classic_families=classic,
        baseline_strong_families=strong,
        to_run_families=to_run,
        to_run_library_ids=to_run_library_ids,
        property_scores=dict(score),
        missing_library_ids=missing_library_ids,
        rationale=rationale,
    )
