"""demo_soft_sense.py — 软测量推理演示。

训练冠军模型（ridge）→ 存权重 → 全时段软测 V 输出 → 单点推理。
用法：PYTHONPATH=src python scripts/demo_soft_sense.py [model_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from boilermind.orchestration.soft_sense import SoftSensor, _load_data, _v31

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"


def _persistence_mae(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.abs(y[1:] - y[:-1]))) if len(y) > 1 else float("nan")


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "ridge"
    sensor = SoftSensor(model_id, dataset_path=DATASET)
    print(f"== 训练 {model_id} 软测器 ==", flush=True)
    sensor.fit()
    print(f"  选参: {sensor.selected_params}", flush=True)
    print(f"  validation MAE: {sensor.validation_mae:.4f}", flush=True)
    print(f"  权重已存: {sensor.weights_path}", flush=True)

    data = _load_data(DATASET)
    Xva, yva = data["X_validation"], data["y_validation"]
    Xte, yte = data["X_locked_test"], data["y_locked_test"]
    pva = sensor.predict(Xva)
    pte = sensor.predict(Xte)
    per_val = _persistence_mae(yva)
    per_locked = _persistence_mae(yte)
    v31 = _v31()

    print(f"\n== 全时段软测评估 ==", flush=True)
    print(f"  validation : MAE={v31.metrics(yva, pva)['mae_m3_s']:.4f} "
          f"(persistence {per_val:.4f})", flush=True)
    print(f"  locked_test: MAE={v31.metrics(yte, pte)['mae_m3_s']:.4f} "
          f"(persistence {per_locked:.4f})", flush=True)

    print(f"\n== 软测 V 序列（validation 前 8 个点）==", flush=True)
    print(f"  {'时刻':>8} | {'实际V':>8} | {'软测V':>8} | {'误差':>8}", flush=True)
    for i in range(8):
        j = i + 100
        print(f"  {j:8d} | {yva[j]:8.4f} | {pva[j]:8.4f} | {abs(yva[j]-pva[j]):8.4f}", flush=True)

    print(f"\n== 单点软测推理 ==", flush=True)
    row = Xva[200].tolist()
    v = sensor.soft_sense_row(row)
    print(f"  给定 31 特征 -> 软测 V = {v:.4f} m3/s (实际 {yva[200]:.4f})", flush=True)

    # 落盘全时段软测序列
    out = ROOT / "runtime" / "soft_sensor"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / f"soft_sense_{model_id}_validation.csv",
               np.column_stack([yva, pva]), delimiter=",", header="actual,soft", comments="")
    np.savetxt(out / f"soft_sense_{model_id}_locked.csv",
               np.column_stack([yte, pte]), delimiter=",", header="actual,soft", comments="")
    print(f"\n  软测序列已存: {out / f'soft_sense_{model_id}_validation.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
