from pathlib import Path
import joblib

path = Path(
    r"D:\BoilerMind-Trusted\resources\models\steam_volume_v1\ridge_if97_mass_flow_model.joblib"
)

obj = joblib.load(path)

print("ARTIFACT_TYPE =", type(obj))

if isinstance(obj, dict):
    print("KEYS =", list(obj.keys()))
    print("MODEL_ID =", obj.get("model_id"))
    print("PREDICTION_MODE =", obj.get("prediction_mode"))
    print("PARAMS =", obj.get("params"))
    print("STATUS =", obj.get("status"))

    features = obj.get("feature_names", [])
    print("FEATURE_COUNT =", len(features))
    print("FIRST_15_FEATURES =", features[:15])
    print("LAST_15_FEATURES =", features[-15:])

    model = obj.get("model")
    print("MODEL_TYPE =", type(model))
    print("N_FEATURES_IN =", getattr(model, "n_features_in_", None))
else:
    print("Unexpected artifact format")
