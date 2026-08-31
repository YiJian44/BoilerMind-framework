from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import MinMaxScaler

REPO = Path(__file__).resolve().parent.parent
DATASET_SHA256 = "9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c"
HORIZONS = (40, 80)
MODELS = ("persistence", "ridge", "bayesianridge")
BLOCKS = (
    ("early", 0.50, 0.60, 0.70),
    ("middle", 0.60, 0.70, 0.80),
    ("late", 0.70, 0.80, 0.90),
    ("latest_holdout", 0.70, 0.80, 1.00),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae_m3_s": float(np.mean(np.abs(error))),
        "rmse_m3_s": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / denominator),
        "mbe_m3_s": float(np.mean(error)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def indices(count: int, train_end: float, validation_end: float, test_end: float, name: str):
    train_stop = int(count * train_end)
    validation_stop = int(count * validation_end)
    if name == "latest_holdout":
        test_start = int(count * 0.90)
    else:
        test_start = validation_stop
    test_stop = int(count * test_end)
    return np.arange(train_stop), np.arange(train_stop, validation_stop), np.arange(test_start, test_stop)


def main() -> int:
    parser = argparse.ArgumentParser(description="BM-TIME-01 leakage-safe expanding-window linear baseline study.")
    parser.add_argument("--cache", default=str(REPO / "runtime/31v_data"))
    parser.add_argument("--out", default=str(REPO / "runtime/experiment_artifacts/BM-TIME-01"))
    parser.add_argument("--report", default=str(REPO / "docs/BM-TIME-01_跨时间块复验报告.md"))
    args = parser.parse_args()
    cache, out, report = Path(args.cache).resolve(), Path(args.out).resolve(), Path(args.report).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_rows, prediction_rows = [], []
    audit = {
        "schema_version": "boilermind.bm_time_01.v1", "experiment_id": "BM-TIME-01",
        "status": "PHASE1_COMPLETED_EXPLORATORY", "dataset_sha256": DATASET_SHA256,
        "models": list(MODELS), "horizons": list(HORIZONS),
        "leakage_control": "original_features_recovered_then_scaler_refit_on_each_block_train_only",
        "blocks": {},
    }
    for horizon in HORIZONS:
        npz_path, scaler_path = cache / f"h{horizon}.npz", cache / f"h{horizon}_scaler.joblib"
        with np.load(npz_path, allow_pickle=True) as data:
            scaled_x, y = np.asarray(data["X"], float), np.asarray(data["y"], float)
            y_source = np.asarray(data["y_source"], float)
            source = np.asarray(data["source_indices"], int)
            target = np.asarray(data["target_indices"], int)
        old_scaler = joblib.load(scaler_path)
        raw_x = old_scaler.inverse_transform(scaled_x.reshape(-1, scaled_x.shape[-1])).reshape(scaled_x.shape)
        count = len(y)
        for block_name, train_end, validation_end, test_end in BLOCKS:
            train_idx, validation_idx, test_idx = indices(count, train_end, validation_end, test_end, block_name)
            scaler = MinMaxScaler().fit(raw_x[train_idx].reshape(-1, raw_x.shape[-1]))
            x_train = scaler.transform(raw_x[train_idx].reshape(-1, raw_x.shape[-1])).reshape(len(train_idx), -1)
            x_validation = scaler.transform(raw_x[validation_idx].reshape(-1, raw_x.shape[-1])).reshape(len(validation_idx), -1)
            x_test = scaler.transform(raw_x[test_idx].reshape(-1, raw_x.shape[-1])).reshape(len(test_idx), -1)
            fitted = {}
            best_alpha, best_mae = None, float("inf")
            for alpha in (0.01, 0.1, 1.0, 10.0):
                candidate = Ridge(alpha=alpha).fit(x_train, y[train_idx])
                mae = float(np.mean(np.abs(candidate.predict(x_validation) - y[validation_idx])))
                if mae < best_mae:
                    fitted["ridge"], best_alpha, best_mae = candidate, alpha, mae
            fitted["bayesianridge"] = BayesianRidge().fit(x_train, y[train_idx])
            predictions = {
                "persistence": y_source[test_idx],
                "ridge": fitted["ridge"].predict(x_test),
                "bayesianridge": fitted["bayesianridge"].predict(x_test),
            }
            block_metrics = {}
            persistence_mae = calculate(y[test_idx], predictions["persistence"])["mae_m3_s"]
            for model, predicted in predictions.items():
                current = calculate(y[test_idx], np.asarray(predicted))
                row = {
                    "experiment_id": "BM-TIME-01", "horizon_steps": horizon, "time_block": block_name,
                    "model": model, "train_count": len(train_idx), "validation_count": len(validation_idx),
                    "test_count": len(test_idx), **current,
                    "mae_improvement_vs_persistence_pct": (persistence_mae - current["mae_m3_s"]) / persistence_mae * 100,
                    "selected_alpha": best_alpha if model == "ridge" else "",
                }
                result_rows.append(row)
                block_metrics[model] = row
                for position, actual, prediction in zip(test_idx, y[test_idx], predicted):
                    prediction_rows.append({
                        "horizon_steps": horizon, "time_block": block_name, "model": model,
                        "source_index": int(source[position]), "target_index": int(target[position]),
                        "y_true": float(actual), "y_pred": float(prediction),
                    })
            audit["blocks"][f"h{horizon}_{block_name}"] = {
                "train_source_index": [int(source[train_idx[0]]), int(source[train_idx[-1]])],
                "validation_source_index": [int(source[validation_idx[0]]), int(source[validation_idx[-1]])],
                "test_source_index": [int(source[test_idx[0]]), int(source[test_idx[-1]])],
                "selected_ridge_alpha": best_alpha,
                "winner": min(block_metrics, key=lambda model: block_metrics[model]["mae_m3_s"]),
            }
    metrics_path, predictions_path = out / "time_block_metrics.csv", out / "time_block_predictions.csv"
    write_csv(metrics_path, result_rows)
    write_csv(predictions_path, prediction_rows)
    audit["metrics_sha256"], audit["predictions_sha256"] = sha256(metrics_path), sha256(predictions_path)
    audit_path = out / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# BM-TIME-01 跨时间块复验报告", "",
        "状态：`PHASE1_COMPLETED_EXPLORATORY`。第一阶段只运行Persistence、Ridge、Bayesian Ridge，seed固定42。", "",
        "每个时间块均从缓存逆变换恢复原始特征，再只用该块训练段重新拟合MinMaxScaler；没有复用未来时间段拟合的缩放器。", "",
        "## 时间块结果", "",
        "| Horizon | Block | Best model | Best MAE | Ridge MAE | Bayesian Ridge MAE | Persistence MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        for block_name, *_ in BLOCKS:
            rows = [r for r in result_rows if r["horizon_steps"] == horizon and r["time_block"] == block_name]
            by_model = {r["model"]: r for r in rows}
            best = min(rows, key=lambda row: row["mae_m3_s"])
            lines.append(f"| h{horizon} | {block_name} | {best['model']} | {best['mae_m3_s']:.6f} | {by_model['ridge']['mae_m3_s']:.6f} | {by_model['bayesianridge']['mae_m3_s']:.6f} | {by_model['persistence']['mae_m3_s']:.6f} |")
    lines += ["", "## 边界", "", "- 这是BM-TIME-01第一阶段，只验证确定性线性基线和Persistence。", "- Torch与RF是否扩展，依据本阶段是否出现明显时间块反转决定。", "- latest_holdout使用与late相同训练/验证窗口，但测试90%-100%末段，用于分离locked-test后半段漂移。", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
