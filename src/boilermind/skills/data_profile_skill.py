"""data_profile_skill.py — 锅炉数据的 6 项数据属性分析（实验问题第一步）。

属性：时序性 / 非线性 / 非高斯 / 奇异值(离群) / 稀疏化 / 降维。
输入：resources/datasets/boiler_181var_v1/boiler_181var_clean.csv
      （181 列带表头，列名 "1".."181"，31特征→V direct 口径）。
输出：DataProfile（结构化画像）+ ModelSelectionPlan（属性→模型库选型）。

实现移植自父项目 delivery_数据属性与软测量/analyze_data_profile.py，
加载层改为 BoilerMind 仓库的 181-var CSV；31 特征列取
DirectVolume31VCapabilityRegistry.FEATURE_COLUMNS 为单一事实源。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from boilermind.core.contracts.data_profile import (
    DataProfile,
    DataProfileMeta,
    DataPropertyProfile,
)
from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
)

from .base import BaseSkill
from .profile_mapper import profile_to_model_selection

# 分析常量（与父包保持一致，保证可复现）
SUBSAMPLE = 8000
TRAIN_FRAC = 0.8
TOP_FEATURES = 30
NEAR_DUP = 0.98
MAD_Z = 3.5
SAMPLING_SECONDS = 15

# 物理列（与 scripts/v31_common.py 一致）
MASS_COL = 16      # 主蒸汽质量流量 t/h
PRESSURE_COL = 1   # 主汽压力 MPa
TEMPERATURE_COL = 9  # 主汽温度 degC
LOAD_COL = 6       # 负荷 MW
R_GAS = 0.461526   # kJ/(kg·K)

# 目标定义冻结（软测 V）：理想气体公式，禁止与 IF97 混用
TARGET_DEFINITION_ID = "steam_volumetric_flow_ideal_gas_v1"
TARGET_FORMULA = (
    "V = M*(1000/3600)*R*(T+273.15)/(P*1000), R=0.461526 kJ/(kg·K)（理想气体；"
    "未验证压力参考前禁止与 IF97 结果混用）"
)

PROPERTY_LABELS = {
    "temporal": "时序性",
    "nonlinearity": "非线性",
    "non_gaussian": "非高斯",
    "singular_outlier": "奇异值/离群",
    "sparsity": "稀疏化",
    "dimensionality": "降维",
}


def _default_dataset_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"
    )


def _volume_flow(mass_flow: np.ndarray, pressure: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """V = M*(1000/3600)*R*(T+273.15)/(P*1000)（与 v31_common / unified_runner 一致）。"""
    return mass_flow * (1000.0 / 3600.0) * R_GAS * (temperature + 273.15) / (pressure * 1000.0)


def _lag1_autocorr(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return float("nan")
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================
# 加载
# ============================================================

def _load_stream(dataset_path: Path):
    """读 181-var CSV → arr(N,181)、V、df、phys。列 j(1-based) 对应 arr[:, j-1]。"""
    df = pd.read_csv(dataset_path)
    cols = [str(c) for c in range(1, 182)]
    arr = df.loc[:, cols].astype(float).to_numpy()  # (N,181)
    P = arr[:, PRESSURE_COL - 1]
    T = arr[:, TEMPERATURE_COL - 1]
    M = arr[:, MASS_COL - 1]
    load = arr[:, LOAD_COL - 1]
    V = _volume_flow(M, P, T)
    phys = {"M": M, "P": P, "T": T, "load": load, "V": V}
    return arr, V, df, phys


# ============================================================
# 采样审计（CSV 无独立时间列，按固定 15s 推断）
# ============================================================

def _sampling_audit(df: pd.DataFrame, n: int) -> dict[str, Any]:
    return {
        "note": "CSV 无独立时间列；采样间隔取固定 15s",
        "n": int(n),
        "span_hours": round(n * SAMPLING_SECONDS / 3600.0, 2),
        "n_gaps_non15s": 0,
        "gap_pct": 0.0,
        "mode_interval_s": SAMPLING_SECONDS,
    }


# ============================================================
# 1 时序性
# ============================================================

def _temporal(agg: dict, phys: dict) -> DataPropertyProfile:
    l1 = {k: _lag1_autocorr(phys[k]) for k in phys}
    adf: dict[str, float] = {}
    for k, y in phys.items():
        try:
            adf[k] = float(adfuller(y, maxlag=1)[1])
        except Exception:
            adf[k] = float("nan")
    nonstat = {k: (np.isfinite(p) and p > 0.05) for k, p in adf.items()}
    mean_l1 = float(np.nanmean([l1[k] for k in l1]))
    if mean_l1 >= 0.95 and any(nonstat.values()):
        verdict = "强时序结构（lag1→1 且非平稳）→ 时序模型（LSTM/GRU）合理"
        fam = ["时序模型", "线性回归"]
        cand = ["lstm", "gru", "ridge", "bayesianridge"]
    else:
        verdict = "时序结构中等 → 时序模型与截面模型都值得比"
        fam = ["时序模型", "线性回归"]
        cand = ["lstm", "gru", "ridge", "bayesianridge"]
    return DataPropertyProfile(
        key="temporal",
        label=PROPERTY_LABELS["temporal"],
        verdict=verdict,
        points_to=fam,
        candidate_families=cand,
        indicators={
            "per_col_lag1": {k: round(v, 4) for k, v in l1.items()},
            "all181_lag1_mean": round(float(np.nanmean(agg["lag1"])), 4),
            "adf_pval": {k: (round(v, 4) if np.isfinite(v) else None) for k, v in adf.items()},
            "adf_nonstationary": nonstat,
        },
    )


# ============================================================
# 2 非线性
# ============================================================

def _nonlinearity(arr: np.ndarray, feat0: list[int], V: np.ndarray, phys: dict) -> DataPropertyProfile:
    n = len(arr)
    cut = int(n * TRAIN_FRAC)
    test_idx = np.arange(cut, n)
    rng = np.random.RandomState(0)
    tr_idx = rng.choice(cut, size=min(cut, SUBSAMPLE), replace=False)
    te_idx = rng.choice(test_idx, size=min(len(test_idx), SUBSAMPLE), replace=False)
    cols = feat0 + [MASS_COL - 1]
    corrmat = np.corrcoef(arr.T)
    rows: list[dict] = []
    for c in cols:
        y = arr[:, c]
        if np.nanstd(y) < 1e-9:
            continue
        corrs = corrmat[c]
        feats = [j for j in range(arr.shape[1]) if j != c and abs(corrs[j]) < NEAR_DUP]
        feats = sorted(feats, key=lambda j: -abs(corrs[j]))[:TOP_FEATURES]
        if len(feats) < 5:
            continue
        X = arr[:, feats]
        scl = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = scl.transform(X[tr_idx]), scl.transform(X[te_idx])
        ridge = Ridge(alpha=100.0).fit(Xtr, y[tr_idx])
        hgb = HistGradientBoostingRegressor(
            max_depth=6, max_iter=150, learning_rate=0.1, random_state=0
        ).fit(Xtr, y[tr_idx])
        r2r = r2_score(y[te_idx], ridge.predict(Xte))
        r2h = r2_score(y[te_idx], hgb.predict(Xte))
        rows.append({"col": int(c + 1), "r2_ridge": r2r, "r2_hgb": r2h, "gain": r2h - r2r})
    gains = np.array([r["gain"] for r in rows])
    med = float(np.nanmedian(gains)) if len(gains) else 0.0
    Xv = arr[tr_idx][:, feat0]
    yv = V[tr_idx]
    rho = np.abs([np.corrcoef(arr[tr_idx, j], yv)[0, 1] for j in feat0])
    mi = mutual_info_regression(Xv, yv, n_neighbors=3, random_state=0)
    gap = float(np.nanmedian(mi - rho)) if len(rho) else 0.0
    if med < 0.02:
        verdict = "非线性增益小 → 线性族为主"
        fam, mods = ["线性回归"], ["ridge", "bayesianridge", "pls"]
    elif med <= 0.05:
        verdict = "中度非线性 → 线性 + HGB/MLP"
        fam, mods = ["线性回归", "树/集成"], ["ridge", "hgb", "mlp"]
    else:
        verdict = "强非线性 → 深度模型合理"
        fam, mods = ["线性回归", "树/集成", "深度"], ["hgb", "lstm", "gru"]
    return DataPropertyProfile(
        key="nonlinearity",
        label=PROPERTY_LABELS["nonlinearity"],
        verdict=verdict,
        points_to=fam,
        candidate_families=mods,
        indicators={
            "n_targets": len(rows),
            "gain_median": round(med, 4),
            "gain_min": round(float(np.nanmin(gains)), 4) if len(gains) else None,
            "gain_max": round(float(np.nanmax(gains)), 4) if len(gains) else None,
            "mi_rho_gap_median": round(gap, 4),
            "top_cols": sorted(rows, key=lambda r: -r["gain"])[:8],
        },
    )


# ============================================================
# 3 非高斯
# ============================================================

def _non_gaussian(feat_arr: np.ndarray, phys: dict) -> DataPropertyProfile:
    cols = [
        ("V", phys["V"]), ("M16", phys["M"]), ("P1", phys["P"]),
        ("T9", phys["T"]), ("Load6", phys["load"]),
    ]
    cols += [(f"f{j}", feat_arr[:, i]) for i, j in enumerate(feat0_columns())]
    skews, kurts = {}, {}
    for k, y in cols:
        y = y[np.isfinite(y)]
        skews[k] = float(stats.skew(y))
        kurts[k] = float(stats.kurtosis(y))
    frac_skew = float(np.mean([abs(v) > 0.5 for v in skews.values()]))
    frac_kurt = float(np.mean([v > 4 for v in kurts.values()]))
    try:
        jb_v = stats.jarque_bera(phys["V"])
        jb_p = float(jb_v[1])
    except Exception:
        jb_p = float("nan")
    tail = frac_skew > 0.4 or frac_kurt > 0.2
    if tail:
        verdict = "重尾/非高斯明显 → 用 MAE/Huber 损失、慎用 GPR/BayesianRidge"
        fam, mods = ["树/集成", "深度"], ["hgb", "lstm", "gru"]
    else:
        verdict = "近高斯 → 线性族与高斯假设模型可用"
        fam, mods = ["线性回归"], ["ridge", "bayesianridge", "pls"]
    return DataPropertyProfile(
        key="non_gaussian",
        label=PROPERTY_LABELS["non_gaussian"],
        verdict=verdict,
        points_to=fam,
        candidate_families=mods,
        indicators={
            "skew": {k: round(v, 3) for k, v in skews.items()},
            "kurtosis": {k: round(v, 3) for k, v in kurts.items()},
            "frac_|skew|>0.5": round(frac_skew, 3),
            "frac_kurt>4": round(frac_kurt, 3),
            "jarquebera_V_p": round(jb_p, 5) if np.isfinite(jb_p) else None,
        },
    )


def feat0_columns() -> list[int]:
    """31 特征 1-based 列（单一事实源）。"""
    return list(DirectVolume31VCapabilityRegistry.FEATURE_COLUMNS)


# ============================================================
# 4 奇异值/离群
# ============================================================

def _singular_outlier(arr: np.ndarray, feat0: list[int], ntrain: int) -> DataPropertyProfile:
    X = arr[:ntrain][:, feat0]
    Xs = StandardScaler().fit_transform(X)
    s = np.linalg.svd(Xs, compute_uv=False)
    s = s[s > 1e-12]
    cond = float(s[0] / s[-1]) if len(s) else float("inf")
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    eff_rank = int(np.searchsorted(energy, 0.99) + 1)
    rates: dict[int, float] = {}
    for i, j in enumerate(feat0):
        x = arr[:ntrain, feat0[i]]
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if mad < 1e-12:
            rates[j] = 0.0
            continue
        z = np.abs(x - med) / (1.4826 * mad)
        rates[j] = float(np.mean(z > MAD_Z))
    rate_max = float(np.nanmax(list(rates.values())))
    verdict = ("特征高度共线 → Ridge/PLS/深度+归一化" if cond > 1e3 else "无严重共线")
    fam = (["线性回归", "PLS"] if cond > 1e3 else ["线性回归"])
    cand = ["ridge", "bayesianridge", "pls"]
    if rate_max > 0.02:
        verdict += "；存在离群 → 鲁棒模型/HGB"
        fam = fam + ["树/集成"]
        cand = cand + ["hgb"]
    return DataPropertyProfile(
        key="singular_outlier",
        label=PROPERTY_LABELS["singular_outlier"],
        verdict=verdict,
        points_to=list(dict.fromkeys(fam)),
        candidate_families=list(dict.fromkeys(cand)),
        indicators={
            "cond_number": round(cond, 2) if np.isfinite(cond) else None,
            "effective_rank_99": eff_rank,
            "mad_outlier_rate_max": round(rate_max, 4),
        },
    )


# ============================================================
# 5 稀疏化
# ============================================================

def _sparsity(df: pd.DataFrame, arr: np.ndarray) -> DataPropertyProfile:
    missing = df.isna().mean()
    missing_pct = float(missing.max() * 100)
    zero_rate = float(np.mean(arr == 0))
    stds = np.nanstd(arr, axis=0)
    near_const = [int(c) for c in range(1, 182) if not np.isfinite(stds[c - 1]) or stds[c - 1] < 1e-9]
    dup = float(df.duplicated().mean())
    verdict_parts = []
    if near_const:
        verdict_parts.append(f"近常数列 {len(near_const)} 个 → 剔除")
    if missing_pct > 2:
        verdict_parts.append("缺失率>2% → 补插/标注")
    if not verdict_parts:
        verdict_parts.append("无近常数列、缺失可忽略")
    return DataPropertyProfile(
        key="sparsity",
        label=PROPERTY_LABELS["sparsity"],
        verdict="；".join(verdict_parts),
        points_to=["线性回归", "时序模型"],
        candidate_families=["ridge", "bayesianridge", "pls", "lstm", "gru"],
        indicators={
            "missing_max_pct": round(missing_pct, 4),
            "zero_rate": round(zero_rate, 5),
            "near_constant_cols": near_const,
            "duplicate_rate": round(dup, 5),
        },
    )


# ============================================================
# 6 降维
# ============================================================

def _dimensionality(arr: np.ndarray, feat0: list[int], ntrain: int) -> DataPropertyProfile:
    X = arr[:ntrain][:, feat0]
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    C = np.corrcoef(Xs.T)
    Cabs = np.abs(C)
    np.fill_diagonal(Cabs, 0)
    if k90 < 5:
        verdict = "内蕴维低 → PLS/Ridge/BayesianRidge 优先"
        fam, mods = ["PLS", "线性回归"], ["pls", "ridge", "bayesianridge"]
    elif k90 > 15:
        verdict = "内蕴维高 → 需容量，深度模型有理"
        fam, mods = ["线性回归", "深度"], ["lstm", "gru", "dlinear", "ridge"]
    else:
        verdict = "内蕴维中等 → 线性族与深度都值得比"
        fam, mods = ["线性回归", "深度"], ["pls", "ridge", "bayesianridge"]
    return DataPropertyProfile(
        key="dimensionality",
        label=PROPERTY_LABELS["dimensionality"],
        verdict=verdict,
        points_to=fam,
        candidate_families=mods,
        indicators={
            "k_for_90pct": k90,
            "k_for_95pct": k95,
            "corr_max_abs": round(float(Cabs.max()), 3),
            "corr_mean_abs": round(float(Cabs.mean()), 3),
            "corr_frac_gt0.9": round(float(np.mean(Cabs > 0.9)), 4),
        },
    )


# ============================================================
# DataProfileSkill
# ============================================================

class DataProfileSkill(BaseSkill):
    """分析锅炉数据的 6 项数据属性，并映射到模型库选型。"""

    name = "data_profile"
    description = "分析锅炉数据 6 项属性（时序/非线性/非高斯/奇异离群/稀疏/降维）并映射到模型库选型"

    @staticmethod
    def compute(
        dataset_path: str | Path | None = None,
        *,
        horizon_steps: int = 0,
        profile_id: str | None = None,
        data_split: str = "all",
    ) -> DataProfile:
        """确定性画像计算（不依赖运行时上下文）。

        data_split：'all' 全量（报告/审计用）| 'train' 仅训练段（选型用，防泄漏）
        | 'validation' | 'locked_test'。
        """
        path = Path(dataset_path) if dataset_path else _default_dataset_path()
        if not path.exists():
            raise ValueError(f"data_profile_dataset_not_found:{path}")
        arr, V, df, phys = _load_stream(path)
        n = len(arr)
        if n < 100:
            raise ValueError(f"data_profile_insufficient_samples:{n}")
        feat0 = [c - 1 for c in feat0_columns()]

        # 分时段切片：selection 画像只用 train（避免候选模型受验证/测试信息影响）
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)
        if data_split in {"train", "validation", "locked_test"}:
            if data_split == "train":
                sl = slice(0, n_train)
            elif data_split == "validation":
                sl = slice(n_train, n_train + n_val)
            else:
                sl = slice(n_train + n_val, n)
            arr = arr[sl]
            V = V[sl]
            df = df.iloc[sl].reset_index(drop=True)
            phys = {k: v[sl] for k, v in phys.items()}
            n = len(arr)

        agg = {"lag1": np.array([_lag1_autocorr(arr[:, j]) for j in range(arr.shape[1])])}
        ntrain = int(n * 0.7)
        feat_arr = arr[:, feat0]

        properties = {
            "temporal": _temporal(agg, phys),
            "nonlinearity": _nonlinearity(arr, feat0, V, phys),
            "non_gaussian": _non_gaussian(feat_arr, phys),
            "singular_outlier": _singular_outlier(arr, feat0, ntrain),
            "sparsity": _sparsity(df, arr),
            "dimensionality": _dimensionality(arr, feat0, ntrain),
        }

        meta = DataProfileMeta(
            dataset_id="boiler_181var_v1",
            dataset_sha256=_sha256(path),
            n_rows=n,
            n_cols=int(arr.shape[1]),
            n_features=len(feat0),
            sampling_interval_seconds=SAMPLING_SECONDS,
            window_steps=20,
            horizon_steps=horizon_steps,
            target_definition_id=TARGET_DEFINITION_ID,
            target_formula=TARGET_FORMULA,
            v_target={
                "mean": round(float(V.mean()), 4),
                "std": round(float(V.std()), 4),
                "range": round(float(V.max() - V.min()), 4),
                "lag1": round(float(_lag1_autocorr(V)), 4),
            },
            sampling_audit=_sampling_audit(df, n),
        )

        return DataProfile(
            profile_id=profile_id or f"PROFILE-{uuid4().hex[:12]}",
            profile_version="1.0",
            computed_at=datetime.now(timezone.utc).isoformat(),
            meta=meta,
            properties=properties,
        )

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """skill 入口：读 context 的 dataset_path / horizon，产出画像 + 选型计划。"""
        dataset_path = context.get("dataset_path")
        horizon_steps = int(context.get("prediction_horizon_steps") or 0)
        profile = self.compute(dataset_path, horizon_steps=horizon_steps)
        plan = profile_to_model_selection(
            profile,
            horizon_steps=horizon_steps,
        )

        # 落盘可审计（runtime/data_profile/profile_<sha12>.json）
        try:
            out_dir = (
                Path(__file__).resolve().parents[3]
                / "runtime" / "data_profile"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"profile_{profile.meta.dataset_sha256[:12]}.json").write_text(
                json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

        return {
            "data_profile": profile.model_dump(mode="json"),
            "profile_model_selection": plan.model_dump(mode="json"),
            "data_profile_required": True,
            "profile": profile,
            "model_selection_plan": plan,
        }
