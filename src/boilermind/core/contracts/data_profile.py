"""数据属性画像与模型选型契约。

画像属性：时序性 / 非线性 / 非高斯 / 奇异值(离群) / 稀疏化 / 降维。
选型计划：数据属性 → 模型库 14 家族 → 具体模型 id。
"""
from __future__ import annotations

from typing import Any

from .base import ContractModel


class DataProfileMeta(ContractModel):
    """画像元信息：数据集、采样、窗口、时域、目标 V 统计、目标定义。"""

    dataset_id: str
    dataset_sha256: str
    n_rows: int
    n_cols: int
    n_features: int
    sampling_interval_seconds: int
    window_steps: int
    horizon_steps: int
    target_definition_id: str = ""
    target_formula: str = ""
    v_target: dict[str, float | int]
    sampling_audit: dict[str, Any]


class DataPropertyProfile(ContractModel):
    """单属性画像：判定 + 指向模型族 + 全部数值指标。"""

    key: str
    label: str
    verdict: str
    points_to: list[str]
    candidate_families: list[str]
    indicators: dict[str, Any]


class DataProfile(ContractModel):
    """完整数据属性画像（6 项属性）。"""

    profile_id: str
    profile_version: str
    computed_at: str
    meta: DataProfileMeta
    properties: dict[str, DataPropertyProfile]


class ModelSelectionPlan(ContractModel):
    """数据属性 → 模型库选型计划（喂给逐模型候选假设生成）。"""

    selection_id: str
    profile_id: str
    horizon_steps: int
    recommended_families: list[str]
    baseline_classic_families: list[str]
    baseline_strong_families: list[str]
    to_run_families: list[str]
    to_run_library_ids: list[str]
    property_scores: dict[str, int]
    missing_library_ids: list[str]
    rationale: list[str]
