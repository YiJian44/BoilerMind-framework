"""数据属性执行器的纯函数测试（复合判据 / 持恒基线，不训练模型）。"""
from __future__ import annotations

import numpy as np

from boilermind.orchestration.data_profile_runner import (
    _composite_gate,
    _persistence_mae,
)


def test_persistence_mae_first_difference():
    y = np.array([1.0, 1.02, 1.01, 1.03, 1.02])
    expected = float(np.mean(np.abs(np.diff(y))))
    assert abs(_persistence_mae(y) - expected) < 1e-12


def test_composite_gate_default_margin():
    # 增益 + 幅度10% + 泛化全过
    ok, reasons = _composite_gate(
        val_mae=0.08, baseline_val_mae=0.10,
        locked_mae=0.05, baseline_locked_mae=0.08,
    )
    assert ok and not reasons

    # 幅度失败（仅 5%）
    ok, reasons = _composite_gate(
        val_mae=0.096, baseline_val_mae=0.10,
        locked_mae=0.05, baseline_locked_mae=0.08,
    )
    assert not ok
    assert any("幅度" in r for r in reasons)

    # 增益失败
    ok, reasons = _composite_gate(
        val_mae=0.11, baseline_val_mae=0.10,
        locked_mae=0.05, baseline_locked_mae=0.08,
    )
    assert not ok
    assert any("增益" in r for r in reasons)

    # 泛化失败
    ok, reasons = _composite_gate(
        val_mae=0.08, baseline_val_mae=0.10,
        locked_mae=0.09, baseline_locked_mae=0.08,
    )
    assert not ok
    assert any("泛化" in r for r in reasons)


def test_composite_gate_relaxed():
    # margin=0：只要求优于基线（幅度退化）
    ok, _ = _composite_gate(
        val_mae=0.099, baseline_val_mae=0.10,
        locked_mae=0.05, baseline_locked_mae=0.08,
        margin=0.0,
    )
    assert ok

    # locked_factor=0：跳过泛化
    ok, _ = _composite_gate(
        val_mae=0.08, baseline_val_mae=0.10,
        locked_mae=0.20, baseline_locked_mae=0.08,
        locked_factor=0.0,
    )
    assert ok
