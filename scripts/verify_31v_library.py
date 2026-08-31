"""Load-check the registered 31/V model library weights (sklearn + torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v31_common import build_deep_module  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "model_library"


def load_check(model_id: str, horizon: int) -> None:
    entry = next(
        m for m in json.loads((LIB / "model_library.json").read_text(encoding="utf-8"))["models"]
        if m["id"] == model_id
    )
    w = entry["weights"]
    mdir = LIB / w["dir"]
    npz = np.load(REPO / "runtime" / "31v_data" / f"h{horizon}.npz", allow_pickle=True)
    idx = npz["locked_test_idx"][-5:]
    X = npz["X"][idx]
    y_true = npz["y"][idx]
    kind = entry["kind"] if "kind" in entry else w["weight_file"].split(".")[-1]

    if (mdir / w["weight_file"]).suffix == ".joblib":
        import joblib

        est = joblib.load(mdir / w["weight_file"])
        pred = est.predict(X.reshape(len(X), -1))
        print(
            f"  [{model_id}] sklearn load OK | pred={np.round(pred,3)} "
            f"true={np.round(y_true,3)} | mean_abs_err={np.mean(np.abs(pred-y_true)):.4f}"
        )
    else:
        import torch

        arch = model_id.rsplit("_h", 1)[0]
        sd = torch.load(mdir / w["weight_file"], map_location="cpu")
        scaler = json.loads((mdir / w["scaler"]).read_text(encoding="utf-8"))
        model = build_deep_module(arch, 31, 20, "cpu")
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            pred = (
                model(torch.from_numpy(X.astype(np.float32))).numpy().reshape(-1)
            ) * scaler["scale"] + scaler["mean"]
        print(
            f"  [{model_id}] torch load OK | pred={np.round(pred,3)} "
            f"true={np.round(y_true,3)} | mean_abs_err={np.mean(np.abs(pred-y_true)):.4f}"
        )


def main() -> int:
    checks = {
        40: ("ridge_h40", "hgb_h40", "lstm_h40", "gru_h40", "transformer_h40"),
        80: ("elasticnet_h80", "pls_h80", "dlinear_h80"),
    }
    failures: list[tuple[str, str]] = []
    for horizon, model_ids in checks.items():
        print(f"== h{horizon} ==")
        for model_id in model_ids:
            try:
                load_check(model_id, horizon)
            except Exception as exc:
                failures.append((model_id, f"{type(exc).__name__}: {exc}"))
                print(f"  [{model_id}] LOAD FAILED | {type(exc).__name__}: {exc}")
    print(f"SUMMARY passed={sum(len(v) for v in checks.values()) - len(failures)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
