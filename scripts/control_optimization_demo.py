"""control_optimization_demo.py — 单条调参假设（含调参范围）生成 + HGB 验证 + Unity 推送。

流程：
  1. 训练 V=f(给煤col5,给水col14,送风col17,汽包压力col2)（HGB）
  2. 随机搜索：在压力≤23MPa 下找能使 V 升 15% 的调整组合
  3. 收敛出一组"可行调整范围"（各变量 min~max）
  4. HGB 按范围软测 V：若预测上升≥15% → 小模型验证成功
  5. 生成单条假设（内含调参范围）→ 输出 Unity 推送 payload（adjustment_ranges）

⚠️ 相关性非因果；列号为候选推断。用法：PYTHONPATH=src python scripts/control_optimization_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"

COLUMNS = ["5", "14", "17", "2"]   # 给煤/给水/送风/汽包压力（候选列）
VAR_NAMES = ["给煤", "给水", "送风", "汽包压力"]
PRESSURE_LIMIT = 23.0               # 压力限制 MPa
TARGET_RISE = 0.15                  # V 升 15%
MAX_ADJ_PCT = 0.25                  # 每变量最大调整 ±25%
N_SAMPLES = 8000


def load_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("v31_common", ROOT / "scripts" / "v31_common.py")
    v31 = importlib.util.module_from_spec(spec)
    sys.modules["v31_common"] = v31
    spec.loader.exec_module(v31)
    df = v31.load_181_frame(str(DATASET))
    X = df.loc[:, COLUMNS].to_numpy(float)
    M = df["16"].to_numpy(float); P = df["1"].to_numpy(float); T = df["9"].to_numpy(float)
    V = v31.volume_flow(M, P, T)
    return X, V


def main() -> int:
    X, V = load_data()
    n = len(V); cut = int(n * 0.7); cutv = int(n * 0.8)
    Xtr, ytr = X[:cut], V[:cut]
    Xva, yva = X[cut:cutv], V[cut:cutv]
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_depth=6, max_iter=200, learning_rate=0.1, random_state=0)
    model.fit(Xtr, ytr)
    val_mae = float(np.mean(np.abs(model.predict(Xva) - yva)))
    print(f"== V=f(给煤,给水,送风,汽包压力) HGB ==  valMAE={val_mae:.4f}", flush=True)

    # 取低 V 工况点（+15% 才在数据范围内可行）
    low_idx = np.argsort(yva)[:10]
    idx = cut + low_idx[2]
    x0 = X[idx].copy()
    v0 = float(model.predict(x0.reshape(1, -1))[0])
    v_target = v0 * (1 + TARGET_RISE)
    print(f"当前工况: {dict(zip(VAR_NAMES, [round(v,1) for v in x0]))} | V={v0:.4f}", flush=True)
    print(f"目标: V 升 {TARGET_RISE*100:.0f}% -> {v_target:.4f} | 压力限制 {PRESSURE_LIMIT}MPa", flush=True)

    # 随机搜索：收集所有在压力限制内使 V 升 ≥15% 的调整
    rng = np.random.RandomState(0)
    hits = []
    best = None
    for _ in range(N_SAMPLES):
        factor = 1 + rng.uniform(-MAX_ADJ_PCT, MAX_ADJ_PCT, 4)
        d = x0 * factor
        if d[3] > PRESSURE_LIMIT:
            continue
        v = float(model.predict(d.reshape(1, -1))[0])
        rise = v / v0 - 1
        if rise >= TARGET_RISE:
            hits.append(d)
        if best is None or abs(rise - TARGET_RISE) < abs(best[0] - TARGET_RISE):
            best = (rise, d)

    if not hits:
        print(f"[WARN] 在压力≤{PRESSURE_LIMIT}MPa 内未找到能使 V 升 {TARGET_RISE*100:.0f}% 的调整（数据V范围有限）", flush=True)
        return 0

    # 可行调整范围（各变量 min~max）
    hits_arr = np.array(hits)
    ranges = [(float(hits_arr[:, i].min()), float(hits_arr[:, i].max())) for i in range(4)]
    rise_best, d_best = best
    v_best = float(model.predict(d_best.reshape(1, -1))[0])

    print("\n== 可行调参范围（HGB 验证）==", flush=True)
    for name, old, (lo, hi) in zip(VAR_NAMES, x0, ranges):
        print(f"  {name}: 当前{old:.1f} -> 建议范围 [{lo:.1f}, {hi:.1f}]", flush=True)
    print(f"  HGB 按范围软测 V: {v0:.4f} -> {v_best:.4f} (上升 {rise_best*100:.1f}%)", flush=True)

    # 生成单条假设（含调参范围）
    from boilermind.orchestration.control_hypothesis_factory import build_control_hypothesis
    hyp = build_control_hypothesis(
        ranges=ranges,
        current_values=x0.tolist(),
        predicted_rise=rise_best,
        target_rise=TARGET_RISE,
        pressure_limit=PRESSURE_LIMIT,
        problem_id="RP-CTRL",
    )
    print(f"\n== 生成的单条假设 ==", flush=True)
    print(f"  {hyp['hypothesis_statement']}", flush=True)
    print(f"  小模型验证: {hyp['validated_on_small_model']}（预测上升 {rise_best*100:.1f}%）", flush=True)

    # Unity 推送 payload
    payload = {
        "hypothesis": hyp,
        "action": "adjust_by_ranges",
        "adjustment_ranges": hyp["adjustment_ranges"],   # [[lo,hi] x4]
        "variable_order": VAR_NAMES,
        "pressure_limit_mpa": PRESSURE_LIMIT,
        "target_rise": TARGET_RISE,
        "unity_note": "Unity 按 adjustment_ranges 调节各变量，演示后对账软测V是否升15%",
    }
    out_dir = ROOT / "runtime" / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "unity_push.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== Unity 推送 payload 已写 {payload_path} ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
