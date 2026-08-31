from pathlib import Path
import json

NEW = Path(r"D:\BoilerMind-Trusted")
OLD = Path(r"D:\BoilerMindTeamTest\_bm_sync_tmp\boilermind-research-v01")

def show_file(path, start=1, end=None):
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    text = path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    )
    lines = text.splitlines()

    if end is None:
        end = len(lines)

    for n in range(start, min(end, len(lines)) + 1):
        print(f"{n:05d}: {lines[n-1]}")

# --------------------------------------------------
# 1. 新框架接口
# --------------------------------------------------

show_file(
    NEW / "src/boilermind/core/contracts/experiment.py"
)

show_file(
    NEW / "src/boilermind/experiment/test_runner.py"
)

show_file(
    NEW / "src/boilermind/planning/plan_contracts.py"
)

# --------------------------------------------------
# 2. 旧项目真实执行核心
# --------------------------------------------------

show_file(
    OLD / "core/experiment_runner/group_experiment_runner.py",
    650,
    820,
)

show_file(
    OLD / "core/experiment_runner/frozen_dataset_builder.py",
    1,
    330,
)

show_file(
    OLD / "adapters/steam_volume_experiment_adapter.py",
    620,
    730,
)

# --------------------------------------------------
# 3. 检查已经迁移的 Ridge artifact
# --------------------------------------------------

print("\n" + "=" * 100)
print("RIDGE ARTIFACT")
print("=" * 100)

ridge_path = (
    NEW
    / "resources/models/steam_volume_v1/"
      "ridge_if97_mass_flow_model.joblib"
)

try:
    import joblib

    model = joblib.load(ridge_path)

    print("type =", type(model))
    print(
        "n_features_in_ =",
        getattr(model, "n_features_in_", None)
    )
    print(
        "feature_names_in_ =",
        getattr(model, "feature_names_in_", None)
    )

    if hasattr(model, "get_params"):
        params = model.get_params()
        print(
            "important_params =",
            {
                k: params[k]
                for k in params
                if k in {
                    "alpha",
                    "fit_intercept",
                }
            }
        )

except Exception as exc:
    print("RIDGE_LOAD_ERROR =", repr(exc))

# --------------------------------------------------
# 4. 检查 LSTM artifact 结构
# --------------------------------------------------

print("\n" + "=" * 100)
print("LSTM ARTIFACT")
print("=" * 100)

lstm_path = (
    NEW
    / "resources/models/steam_volume_v1/"
      "torch_lstm_30_mass_flow_model.pth"
)

try:
    import torch

    obj = torch.load(
        lstm_path,
        map_location="cpu",
        weights_only=False,
    )

    print("type =", type(obj))

    if isinstance(obj, dict):
        print("keys =", list(obj.keys()))

        for key, value in obj.items():
            if isinstance(
                value,
                (str, int, float, bool, list, tuple, dict)
            ):
                print(
                    key,
                    "=",
                    value
                    if len(str(value)) < 1000
                    else "<large value>"
                )
            else:
                print(
                    key,
                    "type =",
                    type(value)
                )

except Exception as exc:
    print("LSTM_LOAD_ERROR =", repr(exc))

