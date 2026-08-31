"""data_profile_runner.py — 数据属性驱动的模型软测对比执行器（阶段C核心）。

设定（软测量语义）：
  - 目标：用当前时刻 31 特征（已知变量）估计 V（难得到的变量），horizon=0。
  - 输入：当前时刻 31 特征（不堆 20 步窗口——窗口会稀释信号，实测当前时刻更优）。
  - 基线：均值预测（无-V 基线）——软测中 V 不可得，persistence 不作为门槛，
    只作参考列报。

流程：
  1. 逐模型用 v31_common 协议训练（sklearn 网格 validation 选优 / TorchSensor）。
  2. 复合判据（相对均值基线）：增益 + 幅度(默认≥10%) + locked_test 泛化。
  3. 达标 → 冠军池；冠军 = 池中 validation MAE 最小者；池空且 best_effort → 最小者。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from boilermind.experiment.capability_registry import DirectVolume31VCapabilityRegistry

GATE_MARGIN = 0.10  # 幅度门槛：val MAE <= (1-0.10)*baseline = 0.90×
SVR_MAX_TRAIN = 8000
TORCH_MAX_EPOCHS = 100
TORCH_PATIENCE = 15

_V31_CACHE: dict | None = None


def _load_v31_common():
    """加载 scripts/v31_common.py（模型库训练协议：网格/TorchSensor/metrics）。"""
    global _V31_CACHE
    if _V31_CACHE is not None:
        return _V31_CACHE
    path = Path(__file__).resolve().parents[3] / "scripts" / "v31_common.py"
    spec = importlib.util.spec_from_file_location("v31_common", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v31_common"] = mod
    spec.loader.exec_module(mod)
    _V31_CACHE = mod
    return mod


def _build_current_time_data(dataset_path: str | Path):
    """当前时刻 31 特征 → V(t)。X:(N,31)，chrono 70/10/20。"""
    v31 = _load_v31_common()
    df = v31.load_181_frame(str(dataset_path))
    feat_cols = [str(c) for c in v31.SOFT_SENSOR_FEATURES]
    X = df.loc[:, feat_cols].to_numpy(dtype=float)  # (N, 31)
    M = df[str(v31.MASS_COL)].to_numpy(dtype=float)
    P = df[str(v31.PRESSURE_COL)].to_numpy(dtype=float)
    T = df[str(v31.TEMPERATURE_COL)].to_numpy(dtype=float)
    y = v31.volume_flow(M, P, T)
    n = len(y)
    cut = int(n * 0.70)
    cutv = cut + int(n * 0.10)
    return {
        "X_train": X[:cut], "y_train": y[:cut],
        "X_validation": X[cut:cutv], "y_validation": y[cut:cutv],
        "X_locked_test": X[cutv:], "y_locked_test": y[cutv:],
    }


def _persistence_mae(y: np.ndarray) -> float:
    """持恒基线（参考列报）：用 t-1 时刻的 V 预测 t 时刻 → 一阶差分 MAE。"""
    y = np.asarray(y, dtype=float)
    if len(y) <= 1:
        return float("nan")
    return float(np.mean(np.abs(y[1:] - y[:-1])))


def _composite_gate(
    val_mae: float | None,
    baseline_val_mae: float | None,
    locked_mae: float | None,
    baseline_locked_mae: float | None,
    *,
    margin: float = GATE_MARGIN,
    locked_factor: float = 1.0,
    require_gain: bool = True,
) -> tuple[bool, list[str]]:
    """复合判据（相对基线，默认均值基线）：增益 + 幅度(margin) + 泛化(locked_factor)。"""
    reasons: list[str] = []
    if val_mae is None or baseline_val_mae is None:
        reasons.append("模型 validation MAE 缺失，无法判增益")
    elif require_gain and val_mae >= baseline_val_mae:
        reasons.append(
            f"增益失败：val MAE {val_mae:.4f} ≥ 基线 {baseline_val_mae:.4f}"
        )
    elif margin is not None and margin > 0 and val_mae > (1.0 - margin) * baseline_val_mae:
        reasons.append(
            f"幅度失败：val MAE {val_mae:.4f} > {(1.0-margin):.2f}×基线 {baseline_val_mae:.4f}"
        )
    if locked_factor is not None and locked_factor > 0:
        if locked_mae is None or baseline_locked_mae is None:
            reasons.append("模型 locked_test MAE 缺失，无法判泛化")
        elif locked_mae >= locked_factor * baseline_locked_mae:
            reasons.append(
                f"泛化失败：locked MAE {locked_mae:.4f} ≥ {locked_factor}×基线 locked {baseline_locked_mae:.4f}"
            )
    return (len(reasons) == 0), reasons


def _train_sklearn(v31, model_id: str, data: dict, seed: int = 42, n_jobs: int = -1) -> dict[str, Any]:
    """sklearn 网格搜索：train-only validation 选优，最终 train-only 拟合（诚实样本外）。"""
    Xtr_f = data["X_train"].reshape(len(data["X_train"]), -1)
    Xva_f = data["X_validation"].reshape(len(data["X_validation"]), -1)
    Xte_f = data["X_locked_test"].reshape(len(data["X_locked_test"]), -1)
    ytr, yva, yte = data["y_train"], data["y_validation"], data["y_locked_test"]
    cap = SVR_MAX_TRAIN if model_id == "svr" else None
    if cap is not None and len(Xtr_f) > cap:
        Xtr_f, ytr = Xtr_f[:cap], ytr[:cap]
    best = None
    candidates = []
    for params in v31.sklearn_grid(model_id):
        est = v31.build_sklearn_estimator(model_id, params, seed=seed, n_jobs=n_jobs)
        est.fit(Xtr_f, ytr)
        val_mae = v31.metrics(yva, est.predict(Xva_f))["mae_m3_s"]
        candidates.append({"params": params, "validation_mae_m3_s": val_mae})
        if best is None or val_mae < best[0]:
            best = (val_mae, params)
    best_mae, best_params = best
    final = v31.build_sklearn_estimator(model_id, best_params, seed=seed, n_jobs=n_jobs)
    final.fit(Xtr_f, ytr)
    val = v31.metrics(yva, final.predict(Xva_f))
    locked = v31.metrics(yte, final.predict(Xte_f))
    return {
        "kind": "sklearn_grid",
        "val_mae": float(val["mae_m3_s"]),
        "locked_mae": float(locked["mae_m3_s"]),
        "selected_params": best_params,
        "selected_val_mae": float(best_mae),
        "n_candidates": len(candidates),
    }


def _train_torch(v31, model_id: str, data: dict, *, device: str = "cpu", seed: int = 42,
                 max_epochs: int = TORCH_MAX_EPOCHS, patience: int = TORCH_PATIENCE) -> dict[str, Any]:
    """TorchSensor（validation 早停），输入为当前时刻特征序列 (N,1,31)。"""
    Xtr = data["X_train"].reshape(len(data["X_train"]), 1, -1)
    Xva = data["X_validation"].reshape(len(data["X_validation"]), 1, -1)
    Xte = data["X_locked_test"].reshape(len(data["X_locked_test"]), 1, -1)
    ytr, yva, yte = data["y_train"], data["y_validation"], data["y_locked_test"]
    sensor = v31.TorchSensor(model_id, max_epochs=max_epochs, patience=patience,
                             device=device, seed=seed)
    sensor.fit(Xtr, ytr, Xva, yva)
    val = v31.metrics(yva, sensor.predict(Xva))
    locked = v31.metrics(yte, sensor.predict(Xte))
    return {
        "kind": "torch_sensor",
        "val_mae": float(val["mae_m3_s"]),
        "locked_mae": float(locked["mae_m3_s"]),
        "epochs_completed": sensor.epochs_completed,
        "best_epoch": sensor.best_epoch_,
    }


def run_data_profile_experiment(
    *,
    to_run_families: list[str],
    dataset_path: str | Path | None = None,
    seed: int = 42,
    margin: float = GATE_MARGIN,
    locked_factor: float = 1.0,
    require_gain: bool = True,
    best_effort: bool = True,
    max_rounds: int = 1,
    torch_device: str | None = None,
    baseline_kind: str = "mean",
) -> dict[str, Any]:
    """执行数据属性驱动的模型软测对比（当前时刻 31 特征，均值基线）。

    baseline_kind：'mean'（均值，软测默认）/ 'persistence'（一阶差分，仅参考对比用）。
    margin/locked_factor/require_gain：复合判据配置。
    best_effort：冠军池空时退化为最小 val MAE 者（标记 is_best_effort）。
    """
    v31 = _load_v31_common()
    path = Path(dataset_path) if dataset_path else (
        DirectVolume31VCapabilityRegistry.DEFAULT_DATASET_PATH
    )
    data = _build_current_time_data(path)
    device = torch_device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # 基线
    if baseline_kind == "persistence":
        base_val = _persistence_mae(data["y_validation"])
        base_locked = _persistence_mae(data["y_locked_test"])
    else:  # mean（无-V）
        base_val = float(np.mean(np.abs(data["y_validation"] - np.mean(data["y_train"]))))
        base_locked = float(np.mean(np.abs(data["y_locked_test"] - np.mean(data["y_train"]))))
    persistence_val = _persistence_mae(data["y_validation"])
    persistence_locked = _persistence_mae(data["y_locked_test"])

    model_results: dict[str, dict[str, Any]] = {}
    champion_pool: list[str] = []
    for name in to_run_families:
        if name == "persistence":
            continue
        try:
            if name in v31.SKLEARN_MODELS:
                trained = _train_sklearn(v31, name, data, seed=seed)
            elif name in v31.DEEP_MODELS:
                trained = _train_torch(v31, name, data, device=device, seed=seed)
            else:
                model_results[name] = {"fit_success": False, "failure_reason": f"unknown_model:{name}"}
                continue
        except Exception as exc:
            model_results[name] = {"fit_success": False, "failure_reason": f"{type(exc).__name__}:{exc}"}
            continue

        val_mae, locked_mae = trained["val_mae"], trained["locked_mae"]
        passed, reasons = _composite_gate(
            val_mae, base_val, locked_mae, base_locked,
            margin=margin, locked_factor=locked_factor, require_gain=require_gain,
        )
        in_pool = bool(passed and np.isfinite(val_mae))
        if in_pool:
            champion_pool.append(name)
        model_results[name] = {
            "fit_success": True,
            "kind": trained["kind"],
            "validation_mae": val_mae,
            "locked_test_mae": locked_mae,
            "baseline_val_mae": base_val,
            "baseline_locked_mae": base_locked,
            "gain_margin": (
                1.0 - val_mae / base_val if np.isfinite(val_mae) and base_val > 0 else None
            ),
            "criterion_passed": passed,
            "criterion_reasons": reasons,
            "in_champion_pool": in_pool,
            **{k: v for k, v in trained.items() if k in {"selected_params", "selected_val_mae", "n_candidates", "epochs_completed", "best_epoch"}},
        }

    winner = None
    is_best_effort = False
    if champion_pool:
        winner = min(champion_pool, key=lambda name: model_results[name]["validation_mae"])
    elif best_effort:
        fitted = [
            name for name, m in model_results.items()
            if m.get("fit_success") and np.isfinite(m.get("validation_mae"))
        ]
        if fitted:
            winner = min(fitted, key=lambda name: model_results[name]["validation_mae"])
            is_best_effort = True

    return {
        "baseline_kind": baseline_kind,
        "baseline_val_mae": base_val,
        "baseline_locked_mae": base_locked,
        "persistence_val_mae": persistence_val,
        "persistence_locked_mae": persistence_locked,
        "to_run_families": list(to_run_families),
        "model_results": model_results,
        "champion_pool": champion_pool,
        "winner": winner,
        "is_best_effort": is_best_effort,
        "winner_validation_mae": (
            model_results[winner]["validation_mae"] if winner else None
        ),
        "winner_locked_test_mae": (
            model_results[winner]["locked_test_mae"] if winner else None
        ),
    }
