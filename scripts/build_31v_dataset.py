"""Build and cache the 31-feature -> V dataset for h40/h80.

Writes runtime/31v_data/{h40,h80}.npz plus meta.json so local and the Aliyun
server use byte-identical splits/scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v31_common import (  # noqa: E402
    build_dataset,
    load_181_frame,
    volume_flow,
    MASS_COL,
    PRESSURE_COL,
    TEMPERATURE_COL,
    SOFT_SENSOR_FEATURES,
    WINDOW,
)

# Re-export for the runner script without the leading underscore.
import v31_common as common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--horizon", nargs="+", type=int, default=[40, 80])
    parser.add_argument("--out", default=str(REPO_ROOT / "runtime" / "31v_data"))
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"data_not_found:{data_path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = _sha256(data_path)

    # sanity: recompute V stats once
    df = load_181_frame(data_path)
    M = df[str(MASS_COL)].to_numpy(dtype=float)
    P = df[str(PRESSURE_COL)].to_numpy(dtype=float)
    T = df[str(TEMPERATURE_COL)].to_numpy(dtype=float)
    V = volume_flow(M, P, T)
    print(
        f"data: {data_path.name}  rows={len(df)}  sha256={digest[:12]}  "
        f"V_mean={V.mean():.3f}  V_std={V.std():.3f}"
    )

    horizons = args.horizon
    meta = {
        "dataset": str(data_path.resolve()),
        "dataset_sha256": digest,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "features_1based": SOFT_SENSOR_FEATURES,
        "mass_col": MASS_COL,
        "pressure_col": PRESSURE_COL,
        "temperature_col": TEMPERATURE_COL,
        "window": WINDOW,
        "train_ratio": 0.70,
        "validation_ratio": 0.10,
        "v_mean": float(V.mean()),
        "v_std": float(V.std()),
        "horizons": horizons,
    }
    for horizon in horizons:
        ds = build_dataset(data_path, horizon=horizon)
        meta[f"h{horizon}"] = {
            "n_total": ds["n_total"],
            "n_train": int(len(ds["split"]["train"])),
            "n_validation": int(len(ds["split"]["validation"])),
            "n_locked_test": int(len(ds["split"]["locked_test"])),
        }
        npz_path = out / f"h{horizon}.npz"
        np.savez(
            npz_path,
            X=ds["X"],
            y=ds["y"],
            y_source=ds["y_source"],
            source_indices=ds["source_indices"],
            target_indices=ds["target_indices"],
            train_idx=ds["split"]["train"],
            validation_idx=ds["split"]["validation"],
            locked_test_idx=ds["split"]["locked_test"],
        )
        # persist the feature scaler for reuse
        import joblib

        joblib.dump(ds["scaler"], out / f"h{horizon}_scaler.joblib")
        print(
            f"  h{horizon}: total={ds['n_total']} train={meta[f'h{horizon}']['n_train']} "
            f"val={meta[f'h{horizon}']['n_validation']} test={meta[f'h{horizon}']['n_locked_test']} -> {npz_path.name}"
        )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"meta -> {out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
