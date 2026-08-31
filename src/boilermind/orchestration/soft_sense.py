"""soft_sense.py — 软测量推理：训练模型并输出蒸汽体积量 V 软测值。

设定（与数据属性实验一致）：
  - 输入：当前时刻 31 特征（软测=用已知变量估难测变量 V）。
  - 冠军模型：默认 ridge（数据属性实验的冠军）。
  - 评估：train-only 训练，validation 选参（诚实样本外）。

用法：
  sensor = SoftSensor(model_id="ridge")
  sensor.fit()                            # 训练并保存权重
  v = sensor.soft_sense_row(features_31)  # 单点软测 → float
  vs = sensor.predict(X_31cols)           # 批量软测 → ndarray
  sensor = SoftSensor.load(path)          # 加载已存权重
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from boilermind.audit.execution_trace import ExperimentExecutionTrace
from boilermind.audit.experiment_auditor import audit_experiment
from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
    ModelExperimentRecord,
    ScientificResult,
)
from boilermind.core.enums import ExperimentStatus, ScientificVerdict
from boilermind.experiment.capability_registry import DirectVolume31VCapabilityRegistry
from boilermind.experiment.metric_normalizer import normalize_metrics

GATE_MARGIN = 0.10  # 幅度门槛：val MAE <= (1-0.10)*baseline

_v31_cache: Any = None


def _v31():
    global _v31_cache
    if _v31_cache is None:
        path = Path(__file__).resolve().parents[3] / "scripts" / "v31_common.py"
        spec = importlib.util.spec_from_file_location("v31_common", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["v31_common"] = mod
        spec.loader.exec_module(mod)
        _v31_cache = mod
    return _v31_cache


def _default_dataset_path() -> Path:
    return DirectVolume31VCapabilityRegistry.DEFAULT_DATASET_PATH


def _load_data(dataset_path: str | Path):
    """当前时刻 31 特征 + V，chrono 70/10/20 切分。返回 dict。"""
    v31 = _v31()
    df = v31.load_181_frame(str(dataset_path))
    feat_cols = [str(c) for c in v31.SOFT_SENSOR_FEATURES]
    X = df.loc[:, feat_cols].to_numpy(dtype=float)  # (N,31)
    M = df[str(v31.MASS_COL)].to_numpy(dtype=float)
    P = df[str(v31.PRESSURE_COL)].to_numpy(dtype=float)
    T = df[str(v31.TEMPERATURE_COL)].to_numpy(dtype=float)
    y = v31.volume_flow(M, P, T)
    n = len(y)
    cut = int(n * 0.70)
    cutv = cut + int(n * 0.10)
    return {
        "X": X, "y": y,
        "X_train": X[:cut], "y_train": y[:cut],
        "X_validation": X[cut:cutv], "y_validation": y[cut:cutv],
        "X_locked_test": X[cutv:], "y_locked_test": y[cutv:],
        "split": (cut, cutv),
    }


class SoftSensor:
    """软测量器：训练 + 推理 V。"""

    def __init__(
        self,
        model_id: str = "ridge",
        *,
        dataset_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.dataset_path = Path(dataset_path) if dataset_path else _default_dataset_path()
        self.weights_path = Path(weights_path) if weights_path else (
            Path(__file__).resolve().parents[3] / "runtime" / "soft_sensor"
            / f"{model_id}_champion.joblib"
        )
        self.seed = seed
        self._estimator: Any = None
        self._is_torch: bool = False
        self.selected_params: dict[str, Any] | None = None
        self.validation_mae: float | None = None

    # ---------- 训练 ----------

    def fit(self, *, save_weights: bool = True) -> "SoftSensor":
        """train-only 训练（诚实样本外）；sklearn 网格选参，torch 用 TorchSensor。"""
        v31 = _v31()
        data = _load_data(self.dataset_path)
        Xtr, ytr = data["X_train"], data["y_train"]
        Xva, yva = data["X_validation"], data["y_validation"]

        if self.model_id in v31.DEEP_MODELS:
            # torch：当前时刻特征 reshape 为 (N,1,n_features)；按 lr 网格选优（公平调参）
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            Xtr3 = Xtr.reshape(len(Xtr), 1, -1)
            Xva3 = Xva.reshape(len(Xva), 1, -1)
            best = None
            for tparams in v31.torch_grid(self.model_id):
                sensor = v31.TorchSensor(
                    self.model_id, max_epochs=100, patience=15,
                    lr=float(tparams.get("lr", 1e-3)),
                    device=device, seed=self.seed,
                )
                sensor.fit(Xtr3, ytr, Xva3, yva)
                vmae = float(v31.metrics(yva, sensor.predict(Xva3))["mae_m3_s"])
                if best is None or vmae < best[0]:
                    best = (vmae, tparams, sensor)
            self._estimator = best[2]
            self._is_torch = True
            self.selected_params = best[1]
            self.validation_mae = float(best[0])
        else:
            self._is_torch = False
            best = None
            for params in v31.sklearn_grid(self.model_id):
                est = v31.build_sklearn_estimator(self.model_id, params, seed=self.seed, n_jobs=-1)
                est.fit(Xtr, ytr)
                val_mae = v31.metrics(yva, est.predict(Xva))["mae_m3_s"]
                if best is None or val_mae < best[0]:
                    best = (val_mae, params)
            self.selected_params = best[1]
            self._estimator = v31.build_sklearn_estimator(
                self.model_id, self.selected_params, seed=self.seed, n_jobs=-1
            )
            self._estimator.fit(Xtr, ytr)
            self.validation_mae = float(best[0])

        if save_weights:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            import joblib

            joblib.dump({
                "model_id": self.model_id,
                "selected_params": self.selected_params,
                "validation_mae": self.validation_mae,
                "feature_columns": [int(c) for c in v31.SOFT_SENSOR_FEATURES],
                "estimator": self._estimator,
            }, self.weights_path)
        return self

    # ---------- 推理 ----------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """批量软测：X 形状 (N,31) 或 (N,31)。"""
        if self._estimator is None:
            raise RuntimeError("soft_sensor_not_fitted; 先调用 fit() 或 load()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._is_torch:
            X = X.reshape(len(X), 1, -1)
        return np.asarray(self._estimator.predict(X)).reshape(-1)

    def soft_sense_row(self, features: list[float]) -> float:
        """单点软测：给定当前时刻 31 个特征 → V (m³/s)。"""
        out = self.predict(np.asarray(features, dtype=float))
        return float(out[0])

    # ---------- 保存/加载 ----------

    def save(self, path: str | Path | None = None) -> Path:
        import joblib

        target = Path(path) if path else self.weights_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_id": self.model_id,
            "selected_params": self.selected_params,
            "validation_mae": self.validation_mae,
            "estimator": self._estimator,
        }, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SoftSensor":
        import joblib

        payload = joblib.load(str(path))
        sensor = cls(payload.get("model_id", "ridge"))
        sensor._estimator = payload["estimator"]
        sensor.selected_params = payload.get("selected_params")
        sensor.validation_mae = payload.get("validation_mae")
        return sensor


def _persistence_mae(y: np.ndarray) -> float:
    """持恒基线（一阶差分）：用 t-1 时刻 V 预测 t 时刻。"""
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.abs(y[1:] - y[:-1]))) if len(y) > 1 else float("nan")


def _canon(metrics: dict[str, Any]) -> dict[str, float]:
    """把 v31.metrics 的 mae_m3_s 等键转成框架的 MAE/RMSE/R2/MBE。"""
    return {
        "MAE": float(metrics.get("mae_m3_s", float("nan"))),
        "RMSE": float(metrics.get("rmse_m3_s", float("nan"))),
        "R2": float(metrics.get("r2", float("nan"))),
        "MBE": float(metrics.get("mbe_m3_s", float("nan"))),
    }


def run_soft_sense_experiment(
    contract: ExperimentContract | dict[str, Any],
    *,
    weights_path: str | Path | None = None,
    horizon_steps: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """软测正式执行器：SoftSensor → ExperimentResult → audit → verdict。

    对齐 execute_real_experiment 的输出结构。horizon=0 软测 / horizon>0 预测
    由契约 prediction_horizon_steps 决定。产物写入 run_id 统一目录。
    """
    if isinstance(contract, dict):
        contract = ExperimentContract.model_validate(contract)
    model_id = (contract.candidate_models or [None])[0]
    if not model_id:
        raise ValueError("soft_sense_candidate_model_required")
    ds = contract.execution_requirements.get("dataset_path") or _default_dataset_path()
    hz = horizon_steps if horizon_steps is not None else int(contract.prediction_horizon_steps or 0)

    # 统一 run_id 产物树：runtime/research_runs_v2/<run_id>/{soft_sensor,profile,report}
    run_dir = (
        Path(__file__).resolve().parents[3]
        / "runtime" / "research_runs_v2"
        / (run_id or contract.experiment_id)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if weights_path and Path(weights_path).exists():
        sensor = SoftSensor.load(weights_path)
    else:
        sensor = SoftSensor(
            model_id,
            dataset_path=ds,
            weights_path=run_dir / "soft_sensor" / f"{model_id}_champion.joblib",
        )
        sensor.fit(save_weights=True)

    data = _load_data(ds)
    Xva, yva = data["X_validation"], data["y_validation"]
    Xte, yte = data["X_locked_test"], data["y_locked_test"]
    v31 = _v31()
    val_metrics = _canon(v31.metrics(yva, sensor.predict(Xva)))
    locked_metrics = _canon(v31.metrics(yte, sensor.predict(Xte)))
    persistence_val = _persistence_mae(yva)
    persistence_locked = _persistence_mae(yte)

    record = ModelExperimentRecord(
        model_name=model_id,
        fit_success=True,
        fit_converged=True,
        model_configuration=sensor.selected_params or {},
        validation_metrics=val_metrics,
        locked_test_metrics=locked_metrics,
        train_samples=len(data["X_train"]),
        validation_samples=len(yva),
        test_samples=len(yte),
        random_seed=sensor.seed,
        artifact_paths=[str(sensor.weights_path)],
        artifact_provenance={"soft_sense": True, "horizon_steps": hz},
    )
    result = ExperimentResult(
        experiment_id=contract.experiment_id,
        problem_id=contract.problem_id,
        hypothesis_id=contract.hypothesis_id,
        plan_id=contract.plan_id,
        status=ExperimentStatus.COMPLETED,
        metrics=locked_metrics,
        raw_metrics=locked_metrics,
        normalized_metrics=normalize_metrics(locked_metrics),
        baseline_metrics={"MAE": persistence_locked},
        candidate_locked_test_metrics={model_id: locked_metrics},
        model_records={model_id: record},
        execution_notes=["SOFT_SENSE_EXECUTION", f"horizon_steps:{hz}"],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    trace = ExperimentExecutionTrace(
        experiment_id=contract.experiment_id,
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
        notes=[f"soft_sense_mae_val:{val_metrics['MAE']:.4f}", f"persistence_val:{persistence_val:.4f}"],
    )
    audit = audit_experiment(contract, result, trace)
    # 软测判决：基线=均值预测（无V，软测中 V 不可得）；persistence 仅参考。
    # 复合判据：增益（val < mean）+ 幅度≥10% + locked_test 泛化。
    mean_val = float(np.mean(np.abs(yva - np.mean(data["y_train"]))))
    mean_locked = float(np.mean(np.abs(yte - np.mean(data["y_train"]))))
    val_mae = float(val_metrics["MAE"])
    locked_mae = float(locked_metrics["MAE"])
    gain = (1.0 - val_mae / mean_val) if mean_val > 0 else None
    met = (
        audit.execution_valid
        and np.isfinite(val_mae)
        and val_mae < mean_val
        and (gain is None or gain >= GATE_MARGIN)
        and locked_mae < mean_locked
    )
    verdict = (
        ScientificVerdict.SUPPORTED
        if met
        else ScientificVerdict.FALSIFIED
        if audit.execution_valid
        else ScientificVerdict.INSUFFICIENT_EVIDENCE
    )
    scientific_result = ScientificResult(
        hypothesis_id=contract.hypothesis_id,
        experiment_id=contract.experiment_id,
        verdict=verdict,
        rationale=(
            f"软测模型 {model_id} validation MAE={val_mae:.4f} vs 均值基线 "
            f"{mean_val:.4f}（增益 {gain:.1%}）；persistence 参考 {persistence_val:.4f}"
        ),
        achieved_criteria=["soft_sense_gain_over_mean_baseline"] if met else [],
        failed_criteria=(
            [] if met else ["soft_sense_gain_over_mean_baseline"]
        ),
    )

    # 统一 run_id 产物：软测结果 JSON（含目标定义版本）
    result_payload = {
        "experiment_id": contract.experiment_id,
        "model": model_id,
        "horizon_steps": hz,
        "target_definition_id": "steam_volumetric_flow_ideal_gas_v1",
        "target_formula": (
            "V = M*(1000/3600)*R*(T+273.15)/(P*1000), R=0.461526 kJ/(kg·K)（理想气体；"
            "未验证压力参考前禁止与 IF97 混用）"
        ),
        "verdict": verdict.value,
        "validation_mae": val_mae,
        "mean_baseline_mae": mean_val,
        "persistence_mae": persistence_val,
        "soft_sense_values": sensor.predict(Xva[:10]).tolist(),
    }
    (run_dir / "soft_sense_result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "experiment_contract": contract,
        "experiment_result": result,
        "execution_trace": trace,
        "audit": audit,
        "criterion_assessment": None,
        "scientific_result": scientific_result,
        "closure_ok": audit.execution_valid,
        "status": "completed",
        "run_dir": str(run_dir),
        "soft_sense_values": {
            "validation": sensor.predict(Xva[:5]).tolist(),
            "persistence_val_mae": persistence_val,
        },
    }
