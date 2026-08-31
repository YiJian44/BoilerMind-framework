"""Train the 31-feature -> V(m^3/s) direct soft-sensing model library (audited build).

For each horizon in {40, 80} and each benchmark model:
  - validation-only model/config selection
  - locked-test evaluation (never used for selection)
  - artifacts written under model_library/:
      weights/31v_direct/{h40,h80}/<model>/     (model + scaler)
      predictions/31v_direct/{h40,h80}/          (per-sample CSVs)
      logs/31v_direct/{h40,h80}/                 (per-model structured logs)
      manifests/31v_direct/{h40,h80}/manifest.json
      SHA256SUMS.json                            (all artifact files)
  - model_library.json registration

Data is loaded from the cached npz (runtime/31v_data/) when present, otherwise
built on the fly from --data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31_common as common  # noqa: E402
from v31_common import (  # noqa: E402
    DEEP_MODELS,
    SKLEARN_MODELS,
    ALL_MODELS,
    SOFT_SENSOR_FEATURES,
    build_sklearn_estimator,
    build_dataset,
    metrics,
    sklearn_grid,
    TorchSensor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "runtime" / "31v_data"
DEFAULT_DATA = REPO_ROOT / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"
LIB_DIR = REPO_ROOT / "model_library"
WEIGHTS_ROOT = LIB_DIR / "weights" / "31v_direct"
PREDICTIONS_ROOT = LIB_DIR / "predictions" / "31v_direct"
LOGS_ROOT = LIB_DIR / "logs" / "31v_direct"
MANIFESTS_ROOT = LIB_DIR / "manifests" / "31v_direct"

SVR_MAX_TRAIN = 8000

FAMILY = {
    "persistence": "统计基线",
    "ridge": "线性", "bayesianridge": "线性", "elasticnet": "线性",
    "pls": "化学计量学", "svr": "核方法", "rf": "树集成",
    "mlp": "神经网络", "knn": "非参数", "hgb": "树集成",
    "transformer": "Transformer", "lstm": "循环神经网络", "dlinear": "线性/时序",
    "gru": "循环神经网络",
}

# Verified names (resources/datasets/boiler_181var_v1/variable_mapping.json)
KNOWN_NAMES = {
    "1": ("main_steam_pressure", "MPa"),
    "6": ("unit_load", "MW"),
    "9": ("main_steam_temperature", "degC"),
}
MASS_COL = 16
PRESSURE_COL = 1
TEMPERATURE_COL = 9


def out_root() -> Path:
    """Effective artifact base dir (model_library/ by default, or --out-root)."""
    return WEIGHTS_ROOT.parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_dataset(data: str | None, cache: str | Path, horizon: int) -> dict:
    cache = Path(cache)
    npz = cache / f"h{horizon}.npz"
    meta_path = cache / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if npz.is_file():
        z = np.load(npz, allow_pickle=True)
        scaler = None
        scaler_path = cache / f"h{horizon}_scaler.joblib"
        if scaler_path.is_file():
            import joblib

            scaler = joblib.load(scaler_path)
        return {
            "X": z["X"], "y": z["y"], "y_source": z["y_source"],
            "source_indices": z["source_indices"], "target_indices": z["target_indices"],
            "split": {
                "train": z["train_idx"], "validation": z["validation_idx"],
                "locked_test": z["locked_test_idx"],
            },
            "scaler": scaler,
            "dataset_sha256": meta.get("dataset_sha256"),
            "rows": meta.get("rows"), "cols": meta.get("cols"),
            "dataset": meta.get("dataset"),
        }
    if data is None:
        raise SystemExit(f"cached dataset not found at {npz}; run build_31v_dataset.py or pass --data")
    ds = build_dataset(data, horizon=horizon)
    ds["dataset"] = str(Path(data).resolve())
    ds["dataset_sha256"] = _sha256(Path(data))
    frame = common.load_181_frame(data)
    ds["rows"] = len(frame)
    ds["cols"] = frame.shape[1]
    return ds


def load_split(ds: dict, name: str):
    idx = ds["split"][name]
    return ds["X"][idx], ds["y"][idx], ds["y_source"][idx]


def protocol_dict(
    horizon: int,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: str,
    sklearn_n_jobs: int,
    parallel_execution: bool,
) -> dict:
    return {
        "task": "31_features_direct_V_soft_sensing",
        "target": "V (m³/s)",
        "target_formula": "V = M*1000/3600 * R*(T+273.15)/(P*1000), R=0.461526, P absolute MPa",
        "feature_columns_1based": SOFT_SENSOR_FEATURES,
        "mass_col": MASS_COL, "pressure_col": PRESSURE_COL, "temperature_col": TEMPERATURE_COL,
        "window_steps": 20,
        "horizon_steps": horizon,
        "sampling_interval_seconds": 15,
        "train_ratio": 0.70, "validation_ratio": 0.10, "locked_test_ratio": 0.20,
        "feature_scaler": "MinMax_train_only",
        "target_scaler": "zscore_train_only",
        "selection_scope": "validation_only",
        "locked_test_used_for_selection": False,
        "random_seed": seed,
        "sklearn_n_jobs": sklearn_n_jobs,
        "parallel_execution": parallel_execution,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "device": device,
    }


def protocol_sha256(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _save_sklearn(model_id: str, horizon: int, est, scaler) -> Path:
    model_dir = WEIGHTS_ROOT / f"h{horizon}" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(est, model_dir / "model.joblib")
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.joblib")
    return model_dir


def _save_torch(model_id: str, horizon: int, sensor: TorchSensor) -> Path:
    model_dir = WEIGHTS_ROOT / f"h{horizon}" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    torch = __import__("torch")
    torch.save(sensor.model.state_dict(), model_dir / "model.pth")
    (model_dir / "target_scaler.json").write_text(
        json.dumps(
            {
                "method": "zscore_train_only",
                "fit_scope": "train_only",
                "mean": float(sensor.y_mean_),
                "scale": float(sensor.y_std_),
                "prediction_inverse_transformed": True,
                "metrics_computed_in_original_unit": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_dir


def export_predictions(
    model_id: str,
    horizon: int,
    ds: dict,
    pred_va: np.ndarray,
    pred_te: np.ndarray,
    dataset_sha256: str,
    protocol_sha: str,
) -> dict[str, str]:
    pred_dir = PREDICTIONS_ROOT / f"h{horizon}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name, pred in (("validation", pred_va), ("locked_test", pred_te)):
        idx = ds["split"][split_name]
        src = ds["source_indices"][idx]
        tgt = ds["target_indices"][idx]
        yt = ds["y"][idx]
        pred = np.asarray(pred, dtype=float).reshape(-1)
        path = pred_dir / f"{model_id}_{split_name}_predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_id", "split", "source_index", "target_index",
                        "y_true", "y_pred", "error", "abs_error",
                        "dataset_sha256", "protocol_sha256"])
            for s, t, yv, yp in zip(src, tgt, yt, pred):
                err = float(yp) - float(yv)
                w.writerow([model_id, split_name, int(s), int(t),
                            round(float(yv), 6), round(float(yp), 6),
                            round(err, 6), round(abs(err), 6),
                            dataset_sha256, protocol_sha])
        paths[split_name] = str(path.relative_to(out_root()))
    return paths


def train_one(
    model_id: str,
    horizon: int,
    ds: dict,
    *,
    device: str,
    max_epochs: int,
    patience: int,
    max_train: int | None,
    seed: int,
    sklearn_n_jobs: int,
    dataset_sha256: str,
    protocol_sha: str,
) -> dict:
    Xtr, ytr, _ = load_split(ds, "train")
    Xva, yva, ys_va = load_split(ds, "validation")
    Xte, yte, _ = load_split(ds, "locked_test")
    started = datetime.now(timezone.utc).isoformat()
    entry: dict = {"model": model_id, "horizon_steps": horizon, "kind": None,
                   "started_at": started, "fit_success": False}

    if model_id == "persistence":
        pred_va = ys_va
        pred_te = ds["y_source"][ds["split"]["locked_test"]]
        pred_paths = export_predictions(model_id, horizon, ds, pred_va, pred_te,
                                        dataset_sha256, protocol_sha)
        entry.update({
            "kind": "heuristic",
            "validation": metrics(yva, pred_va),
            "locked_test": metrics(yte, pred_te),
            "fit_success": True,
            "predictions": pred_paths,
        })
        return entry

    if model_id in SKLEARN_MODELS:
        Xtr_f = Xtr.reshape(len(Xtr), -1)
        Xva_f = Xva.reshape(len(Xva), -1)
        Xte_f = Xte.reshape(len(Xte), -1)
        cap = SVR_MAX_TRAIN if model_id == "svr" else None
        if max_train is not None and (cap is None or max_train < cap):
            cap = max_train
        if cap is not None and len(Xtr_f) > cap:
            Xtr_f, ytr = Xtr_f[:cap], ytr[:cap]

        candidates = []
        best = None
        for params in sklearn_grid(model_id):
            est = build_sklearn_estimator(
                model_id, params, seed=seed, n_jobs=sklearn_n_jobs
            )
            est.fit(Xtr_f, ytr)
            pred_va = est.predict(Xva_f)
            val_mae = metrics(yva, pred_va)["mae_m3_s"]
            candidates.append({"params": params, "validation_mae_m3_s": val_mae})
            if best is None or val_mae < best[0]:
                best = (val_mae, params)
        best_mae, best_params = best

        Xdev = np.vstack([Xtr_f, Xva_f])
        ydev = np.concatenate([ytr, yva])
        final = build_sklearn_estimator(
            model_id, best_params, seed=seed, n_jobs=sklearn_n_jobs
        )
        final.fit(Xdev, ydev)
        pred_te = final.predict(Xte_f)
        pred_va_final = final.predict(Xva_f)
        final_val = metrics(yva, pred_va_final)
        model_dir = _save_sklearn(model_id, horizon, final, ds["scaler"])
        pred_paths = export_predictions(model_id, horizon, ds, pred_va_final, pred_te,
                                        dataset_sha256, protocol_sha)
        entry.update({
            "kind": "sklearn",
            "effective_random_state": seed if model_id in {"elasticnet", "rf", "mlp", "hgb", "gpr"} else None,
            "sklearn_n_jobs": sklearn_n_jobs if model_id in {"rf", "knn"} else None,
            "candidates": candidates,
            "selected_params": best_params,
            "selected_by_validation_mae_m3_s": best_mae,
            "validation": final_val,
            "locked_test": metrics(yte, pred_te),
            "fit_success": True,
            "weights_dir": str(model_dir.relative_to(out_root())),
            "weight_file": "model.joblib",
            "scaler_file": "scaler.joblib",
            "predictions": pred_paths,
        })
        return entry

    if model_id in DEEP_MODELS:
        if max_train is not None and len(Xtr) > max_train:
            Xtr, ytr = Xtr[:max_train], ytr[:max_train]
        sensor = TorchSensor(model_id, max_epochs=max_epochs, patience=patience,
                             device=device, seed=seed)
        sensor.fit(Xtr, ytr, Xva, yva)
        pred_va = sensor.predict(Xva)
        pred_te = sensor.predict(Xte)
        model_dir = _save_torch(model_id, horizon, sensor)
        pred_paths = export_predictions(model_id, horizon, ds, pred_va, pred_te,
                                        dataset_sha256, protocol_sha)
        entry.update({
            "kind": "torch",
            "effective_random_state": seed,
            "epochs_completed": sensor.epochs_completed,
            "best_epoch": sensor.best_epoch_,
            "best_validation_loss": sensor.best_validation_loss_,
            "training_history": sensor.training_history_,
            "validation": metrics(yva, pred_va),
            "locked_test": metrics(yte, pred_te),
            "fit_success": True,
            "weights_dir": str(model_dir.relative_to(out_root())),
            "weight_file": "model.pth",
            "scaler_file": "target_scaler.json",
            "predictions": pred_paths,
        })
        return entry

    raise ValueError(f"unknown_model:{model_id}")


def write_per_model_logs(horizon: int, entries: list[dict]) -> None:
    log_dir = LOGS_ROOT / f"h{horizon}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "models.json").write_text(
        json.dumps({e["model"]: e for e in entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_manifest(
    horizon: int,
    ds: dict,
    entries: list[dict],
    protocol: dict,
    protocol_sha: str,
    *,
    device: str,
    started_at: str,
    completed_at: str,
) -> None:
    def bounds(name: str) -> dict:
        idx = ds["split"][name]
        return {
            "count": int(len(idx)),
            "source_index_min": int(ds["source_indices"][idx][0]),
            "source_index_max": int(ds["source_indices"][idx][-1]),
            "target_index_min": int(ds["target_indices"][idx][0]),
            "target_index_max": int(ds["target_indices"][idx][-1]),
        }

    features = []
    for col in SOFT_SENSOR_FEATURES:
        name, unit = KNOWN_NAMES.get(str(col), (None, None))
        features.append({"column_1based": col, "name": name, "unit": unit})

    manifest = {
        "schema_version": "boilermind.31v_model_library.manifest.v1",
        "task": protocol["task"],
        "dataset": {
            "file": ds.get("dataset"),
            "sha256": ds.get("dataset_sha256"),
            "rows": ds.get("rows"),
            "cols": ds.get("cols"),
        },
        "features": {"count": len(SOFT_SENSOR_FEATURES), "list": features},
        "target": {
            "type": "V (m³/s)",
            "formula": protocol["target_formula"],
            "mass_col": MASS_COL, "pressure_col": PRESSURE_COL, "temperature_col": TEMPERATURE_COL,
        },
        "window_steps": protocol["window_steps"],
        "horizon_steps": horizon,
        "sampling_interval_seconds": protocol["sampling_interval_seconds"],
        "split": {
            "policy": "chronological_train_validation_locked_test",
            "ratios": {"train": 0.70, "validation": 0.10, "locked_test": 0.20},
            "n_total": int(len(ds["y"])),
            "train": bounds("train"),
            "validation": bounds("validation"),
            "locked_test": bounds("locked_test"),
            "locked_test_used_for_selection": False,
        },
        "scaling": {
            "feature_scaler": "MinMax_train_only (fit on train-origin rows only)",
            "target_scaler": "zscore_train_only (fit on train split only)",
            "feature_scaler_file": f"h{horizon}_scaler.joblib",
        },
        "selection": {"scope": "validation_only", "metric": "mae_m3_s"},
        "protocol_sha256": protocol_sha,
        "random_seed": protocol["random_seed"],
        "sklearn_n_jobs": protocol["sklearn_n_jobs"],
        "parallel_execution": protocol["parallel_execution"],
        "max_epochs": protocol["max_epochs"],
        "early_stopping_patience": protocol["early_stopping_patience"],
        "device": device,
        "git_commit": _git_head(),
        "training_started_at": started_at,
        "training_completed_at": completed_at,
        "models": {e["model"]: {
            "kind": e.get("kind"),
            "fit_success": e.get("fit_success"),
            "selected_params": e.get("selected_params"),
            "effective_random_state": e.get("effective_random_state"),
            "sklearn_n_jobs": e.get("sklearn_n_jobs"),
            "epochs_completed": e.get("epochs_completed"),
            "best_epoch": e.get("best_epoch"),
            "best_validation_loss": e.get("best_validation_loss"),
            "validation": e.get("validation"),
            "locked_test": e.get("locked_test"),
        } for e in entries},
    }
    mdir = MANIFESTS_ROOT / f"h{horizon}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sha256sums() -> None:
    out_root = WEIGHTS_ROOT.parent.parent
    records = []
    for root, _dirs, files in sorted(os.walk(WEIGHTS_ROOT)):
        for name in sorted(files):
            path = Path(root) / name
            records.append({
                "path": str(path.relative_to(out_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    for sub in (PREDICTIONS_ROOT, LOGS_ROOT, MANIFESTS_ROOT):
        if sub.is_dir():
            for path in sorted(sub.rglob("*")):
                if path.is_file():
                    records.append({
                        "path": str(path.relative_to(out_root)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    })
    (out_root / "SHA256SUMS.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def build_library_json(entries: list[dict]) -> dict:
    models = []
    for e in entries:
        if e.get("kind") == "failed":
            continue
        h = e["horizon_steps"]
        mid = f"{e['model']}_h{h}"
        weight_file = e.get("weight_file")
        models.append({
            "id": mid,
            "name": f"{e['model']} (31V direct h{h})",
            "family": FAMILY.get(e["model"], e["model"]),
            "collection": "31v_direct",
            "task": {
                "input_dim": 31,
                "window_steps": 20,
                "horizon_steps": h,
                "horizon_minutes": int(round(h * 15 / 60)),
                "target": "V (m³/s)",
                "target_type": "V (m³/s)",
                "feature_scheme": "31 SOFT_SENSOR_FEATURES (1-based)",
                "note": "31特征→V直接软测量；前视随模型记录",
            },
            "weights": {
                "exists": weight_file is not None,
                "dir": e.get("weights_dir"),
                "weight_file": weight_file,
                "scaler": e.get("scaler_file"),
            },
            "metrics": {"validation": e.get("validation"), "locked_test": e.get("locked_test")},
            "status": "benchmark_active",
        })
    return {
        "library_version": "1.2.0",
        "scope": (
            "31特征→蒸汽体积流量V(m³/s)直接软测量；window=20；前视 h40/h80；"
            "切分70/10/20时序，train-only缩放，validation选模，locked_test评估"
        ),
        "count": len(models),
        "collections": ["31v_direct"],
        "deployment_recommendation": {
            "note": "31V direct 软测量库；切分70/10/20，validation选模，locked_test评估",
            "weights_storage": (
                "权重未入 git（rf joblib 超 GitHub 100MB 限制）。"
                "位置：服务器 /root/31v_train/model_library/weights/ 与本地仓库 model_library/weights/（已 gitignore）；脚本可随时重训"
            ),
            "sklearn_version_lock": "sklearn joblib 权重在 scikit-learn 1.7.2 环境训练，跨版本加载需匹配环境或重训；torch .pth 版本无关",
        },
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", nargs="+", type=int, default=[40, 80])
    parser.add_argument("--models", nargs="+", default=None, help="default = all 14")
    parser.add_argument("--data", default=None)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sklearn-n-jobs", type=int, default=-1,
        help="CPU workers for RF/KNN; use 2 when running three RF seeds in parallel",
    )
    parser.add_argument(
        "--parallel-execution", action="store_true",
        help="record that this run shares host resources with other experiment processes",
    )
    parser.add_argument("--out-json", default=str(LIB_DIR / "model_library.json"))
    parser.add_argument("--out-root", default=str(LIB_DIR),
                        help="artifact base dir (weights/predictions/logs/manifests); defaults to model_library/")
    args = parser.parse_args()

    global WEIGHTS_ROOT, PREDICTIONS_ROOT, LOGS_ROOT, MANIFESTS_ROOT
    out_root = Path(args.out_root)
    WEIGHTS_ROOT = out_root / "weights" / "31v_direct"
    PREDICTIONS_ROOT = out_root / "predictions" / "31v_direct"
    LOGS_ROOT = out_root / "logs" / "31v_direct"
    MANIFESTS_ROOT = out_root / "manifests" / "31v_direct"

    models = args.models or list(ALL_MODELS)
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"unknown_models:{unknown}")
    if args.device not in {"cpu", "cuda"}:
        raise SystemExit("device must be cpu or cuda")
    if args.sklearn_n_jobs == 0 or args.sklearn_n_jobs < -1:
        raise SystemExit("sklearn-n-jobs must be -1 or a positive integer")
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("cuda requested but unavailable")

    all_entries: list[dict] = []
    for horizon in args.horizon:
        print(f"\n=== horizon={horizon} ===", flush=True)
        ds = load_dataset(args.data, args.cache, horizon)
        dataset_sha256 = ds["dataset_sha256"] or ""
        protocol = protocol_dict(
            horizon,
            seed=args.seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=args.device,
            sklearn_n_jobs=args.sklearn_n_jobs,
            parallel_execution=args.parallel_execution,
        )
        proto_sha = protocol_sha256(protocol)
        print(f"dataset_sha256={dataset_sha256[:12]} protocol_sha256={proto_sha[:12]}", flush=True)
        horizon_started = datetime.now(timezone.utc).isoformat()
        entries = []
        for model_id in models:
            t0 = time.perf_counter()
            try:
                e = train_one(model_id, horizon, ds,
                              device=args.device, max_epochs=args.max_epochs,
                              patience=args.patience, max_train=args.max_train,
                              seed=args.seed, sklearn_n_jobs=args.sklearn_n_jobs,
                              dataset_sha256=dataset_sha256,
                              protocol_sha=proto_sha)
                lt = e.get("locked_test", {})
                va = e.get("validation", {})
                print(
                    f"  {model_id:<13} kind={e['kind']:<9} "
                    f"val MAE={va.get('mae_m3_s', float('nan')):.4f} R2={va.get('r2', float('nan')):.3f} | "
                    f"test MAE={lt.get('mae_m3_s', float('nan')):.4f} R2={lt.get('r2', float('nan')):.3f} "
                    f"({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                e = {"model": model_id, "horizon_steps": horizon, "kind": "failed",
                     "fit_success": False, "failure": f"{type(exc).__name__}: {exc}"}
                print(f"  {model_id:<13} FAILED: {exc}", flush=True)
            e["runtime_seconds"] = round(time.perf_counter() - t0, 3)
            entries.append(e)
        horizon_completed = datetime.now(timezone.utc).isoformat()
        write_per_model_logs(horizon, entries)
        write_manifest(horizon, ds, entries, protocol, proto_sha,
                       device=args.device, started_at=horizon_started, completed_at=horizon_completed)
        all_entries.extend(entries)

    write_sha256sums()
    library = build_library_json(all_entries)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out} ({library['count']} entries)")
    print(f"predictions -> {PREDICTIONS_ROOT}")
    print(f"logs -> {LOGS_ROOT}")
    print(f"manifests -> {MANIFESTS_ROOT}")
    print(f"SHA256SUMS -> {LIB_DIR / 'SHA256SUMS.json'}")

    n_fail = sum(1 for r in all_entries if r.get("kind") == "failed")
    print(f"failed: {n_fail}/{len(all_entries)}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
