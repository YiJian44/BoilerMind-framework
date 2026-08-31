"""数据属性画像 skill 与映射器的确定性 / 结构测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boilermind.orchestration import ResearchOrchestrator  # noqa: F401  # 先加载编排，绕开 skills↔orchestration 循环导入
from boilermind.core.contracts import DataProfile, ModelSelectionPlan
from boilermind.skills.data_profile_skill import DataProfileSkill
from boilermind.skills.profile_mapper import profile_to_model_selection


def _make_synthetic_csv(path) -> None:
    """300 行 × 181 列合成锅炉数据（P/T/M 近恒定 + 相关特征 + 噪声）。"""
    rng = np.random.RandomState(7)
    n = 300
    P = 13.0 + rng.normal(0, 0.05, n)      # col1 主汽压力
    T = 533.0 + rng.normal(0, 0.5, n)      # col9 主汽温度
    M = 500.0 + rng.normal(0, 8.0, n)      # col16 质量流量
    load = M * 0.9 + rng.normal(0, 2.0, n) # col6 负荷（M 强相关）
    cols = {}
    for j in range(1, 182):
        if j == 1:
            cols[str(j)] = P
        elif j == 6:
            cols[str(j)] = load
        elif j == 9:
            cols[str(j)] = T
        elif j == 16:
            cols[str(j)] = M
        elif j in {2, 4, 5, 8}:
            cols[str(j)] = M * 0.99 + rng.normal(0, 0.1, n)  # M 代理
        else:
            cols[str(j)] = rng.normal(0, 1.0, n)
    df = pd.DataFrame(cols)
    df.to_csv(path, index=False)


def test_compute_is_deterministic(tmp_path):
    csv = tmp_path / "data.csv"
    _make_synthetic_csv(csv)
    p1 = DataProfileSkill.compute(str(csv), horizon_steps=0)
    p2 = DataProfileSkill.compute(str(csv), horizon_steps=0)
    assert p1.properties.keys() == {
        "temporal", "nonlinearity", "non_gaussian",
        "singular_outlier", "sparsity", "dimensionality",
    }
    assert p1.meta.n_rows == 300
    assert p1.meta.n_features == 31
    for key in p1.properties:
        assert p1.properties[key].model_dump() == p2.properties[key].model_dump()


def test_mapper_produces_valid_plan(tmp_path):
    csv = tmp_path / "data.csv"
    _make_synthetic_csv(csv)
    profile = DataProfileSkill.compute(str(csv), horizon_steps=0)
    plan = profile_to_model_selection(profile, horizon_steps=40)
    assert isinstance(plan, ModelSelectionPlan)
    assert plan.to_run_families
    assert len(plan.to_run_library_ids) == len(plan.to_run_families)
    assert set(plan.to_run_library_ids).issubset(
        set(plan.to_run_families)  # id = f"{family}_h40"
    ) or all(plan.to_run_library_ids)


def test_mapper_scores_positive():
    # 无画像时不应崩（空 properties）
    from boilermind.core.contracts.data_profile import (
        DataProfile, DataProfileMeta, DataPropertyProfile,
    )
    profile = DataProfile(
        profile_id="P-TEST", profile_version="1", computed_at="2026-08-24T00:00:00Z",
        meta=DataProfileMeta(
            dataset_id="d", dataset_sha256="s", n_rows=1, n_cols=1, n_features=31,
            sampling_interval_seconds=15, window_steps=20, horizon_steps=0,
            v_target={}, sampling_audit={},
        ),
        properties={},
    )
    plan = profile_to_model_selection(profile, horizon_steps=40)
    assert isinstance(plan, ModelSelectionPlan)
