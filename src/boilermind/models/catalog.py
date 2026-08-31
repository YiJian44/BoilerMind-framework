"""
Default model catalog, derived from the audited old project
(D:\\BoilerMindTeamTest\\_bm_sync_tmp).

Sources:
- boilermind-research-v01/model_library/model_library.json
  (18 benchmark + 3 legacy)
- model_registry/model_catalog.json
  (legacy mass-flow 1/2/5/10min + direct-volume candidates)
- model_library/weights/psfa/*/config.json
  (PSFA h0 / h20 / h80)

The catalog is metadata only. Current executability is
decided by ExperimentCapabilityRegistry, never here.
"""

from __future__ import annotations

from boilermind.models.model_registry import (
    ModelRegistry,
    ModelSpec,
)


OLD_LIBRARY = (
    r"D:\BoilerMindTeamTest\_bm_sync_tmp"
    r"\boilermind-research-v01\model_library"
)

OLD_MODEL_DIR = (
    r"D:\BoilerMindTeamTest\_bm_sync_tmp"
    r"\boiler_soft_sensor_models"
)


_MASS_FLOW_TARGETS = [
    "main_steam_mass_flow",
    "steam_volumetric_flow_derived",
]

_VOLUME_TARGETS = [
    "steam_volumetric_flow",
]

_METRICS = [
    "MAE",
    "RMSE",
    "R2",
    "MBE",
]

_GENERIC_REGRESSION_TARGETS = [
    "NOx",
    "nox_emission",
]

_OPTIMIZATION_TARGETS = [
    "optimization_objective",
    "boiler_efficiency",
    "coal_feed",
    "air_flow",
]


def _compatible_checkpoint(
    *,
    target_ok: bool = True,
    features_ok: bool = True,
    sampling_ok: bool = True,
    window_ok: bool = True,
    horizon_ok: bool = True,
    normalization_ok: bool = True,
    architecture_ok: bool = True,
    note: str = (
        "target/features/sampling/window/horizon/"
        "normalization/architecture all match the "
        "current 30-feature 20-step 40-step contract."
    ),
) -> dict:

    checks = {
        "target": target_ok,
        "features": features_ok,
        "sampling_interval": sampling_ok,
        "window_steps": window_ok,
        "prediction_horizon": horizon_ok,
        "normalization": normalization_ok,
        "architecture": architecture_ok,
    }

    mismatches = [
        name
        for name, ok in checks.items()
        if not ok
    ]

    return {
        "compatible": not mismatches,
        "checked": list(checks),
        "mismatches": mismatches,
        "note": note,
    }


def _sklearn_spec(
    *,
    name: str,
    family: str,
    uncertainty: bool = False,
    cost: str = "seconds",
    checkpoint_path: str | None = None,
) -> ModelSpec:

    return ModelSpec(
        model_name=name,
        framework="sklearn",
        supported_tasks=[
            "prediction",
            "soft_sensor_prediction",
            "nox_prediction",
            "mass_flow_forecast",
            "steam_volume_forecast",
            "optimization_surrogate",
        ],
        required_input_type="flattened_window",
        required_features=30,
        feature_indices=list(range(30)),
        sequence_required=True,
        window_requirements={
            "steps": 20,
            "lookback_minutes": 5,
            "sampling_interval_seconds": 15,
        },
        horizon_capability={
            "steps": 40,
            "minutes": 10,
            "supported_steps": [40, 80],
        },
        supported_targets=(
            list(_MASS_FLOW_TARGETS)
            + list(_VOLUME_TARGETS)
            + list(_GENERIC_REGRESSION_TARGETS)
            + list(_OPTIMIZATION_TARGETS)
        ),
        training_available=True,
        inference_available=True,
        checkpoint_available=(
            checkpoint_path is not None
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_compatibility=(
            _compatible_checkpoint(
                note=(
                    "MinMax scaler + 20x30 flattened window; "
                    "checkpoint contract matches the current "
                    "real_sklearn backend contract."
                ),
            )
        ),
        supported_metrics=list(_METRICS),
        supports_uncertainty=uncertainty,
        compute_cost=cost,
        status="benchmark_active",
        family=family,
        source=OLD_LIBRARY,
        trainable=True,
        inference_supported=True,
        checkpoint_required=False,
        requires_torch=False,
        requires_cuda=False,
        cpu_supported=True,
        gpu_supported=False,
        source_available=True,
        train_from_source_supported=True,
        checkpoint_inference_supported=True,
        adapter_available=True,
        runner_callable=True,
        required_dependencies=["sklearn"],
        data_contract_compatible=True,
    )


def _deep_spec(
    *,
    name: str,
    family: str,
    uncertainty: bool = False,
    cost: str = "minutes",
    source_code_path: str | None = None,
    adapter_ready: bool = False,
) -> ModelSpec:

    return ModelSpec(
        model_name=name,
        framework="torch",
        supported_tasks=["mass_flow_forecast", "steam_volume_forecast"],
        required_input_type="sequence_window",
        required_features=30,
        feature_indices=list(range(30)),
        sequence_required=True,
        window_requirements={
            "steps": 20,
            "lookback_minutes": 5,
            "sampling_interval_seconds": 15,
        },
        horizon_capability={
            "steps": 40,
            "minutes": 10,
            "supported_steps": [40, 80],
        },
        supported_targets=list(_MASS_FLOW_TARGETS) + list(_VOLUME_TARGETS),
        training_available=adapter_ready,
        inference_available=False,
        checkpoint_available=True,
        checkpoint_path=(
            f"{OLD_LIBRARY}\\weights\\benchmark_deep"
            f"\\{name}\\model.pth"
        ),
        checkpoint_compatibility=(
            _compatible_checkpoint(
                note=(
                    "PackagedModel contract (n_past=20, "
                    "offset=40, 30 features, MinMax scaler, "
                    "M target) matches the current contract; "
                    "inference requires torch + old vendor "
                    "loader (not installed in current venv)."
                ),
            )
        ),
        supported_metrics=list(_METRICS),
        supports_uncertainty=uncertainty,
        compute_cost=cost,
        status="checkpoint_ready",
        family=family,
        source=OLD_LIBRARY,
        source_code_path=source_code_path,
        trainable=True,
        inference_supported=True,
        checkpoint_required=False,
        requires_torch=True,
        requires_cuda=False,
        cpu_supported=True,
        gpu_supported=True,
        source_available=bool(source_code_path),
        train_from_source_supported=adapter_ready,
        checkpoint_inference_supported=True,
        architecture_factory=(name if adapter_ready else None),
        adapter_available=adapter_ready,
        runner_callable=adapter_ready,
        required_dependencies=["torch"],
        data_contract_compatible=True,
        remaining_blocker=(None if adapter_ready else "architecture_factory_not_reviewed"),
    )


def _legacy_spec(
    *,
    name: str,
    family: str,
    framework: str,
    tasks: list[str] | None = None,
    features: int,
    window: dict,
    horizon: dict,
    targets: list[str],
    input_type: str,
    checkpoint_path: str | None,
    compatibility: dict,
    status: str,
    source_code_path: str | None = None,
    trainable: bool = True,
    inference_supported: bool = True,
    uncertainty: bool = False,
    cost: str = "minutes",
) -> ModelSpec:

    return ModelSpec(
        model_name=name,
        framework=framework,
        supported_tasks=(
            tasks or ["mass_flow_forecast"]
        ),
        required_input_type=input_type,
        required_features=features,
        feature_indices=None,
        sequence_required=True,
        window_requirements=window,
        horizon_capability=horizon,
        supported_targets=targets,
        training_available=False,
        inference_available=False,
        checkpoint_available=(
            checkpoint_path is not None
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_compatibility=compatibility,
        supported_metrics=list(_METRICS),
        supports_uncertainty=uncertainty,
        compute_cost=cost,
        status=status,
        family=family,
        source=OLD_MODEL_DIR,
        source_code_path=source_code_path,
        trainable=trainable,
        inference_supported=inference_supported,
        checkpoint_required=False,
        requires_torch=(framework == "torch"),
        requires_cuda=False,
        cpu_supported=True,
        gpu_supported=(framework == "torch"),
        source_available=bool(source_code_path),
        train_from_source_supported=False,
        checkpoint_inference_supported=inference_supported,
        adapter_available=False,
        runner_callable=False,
        required_dependencies=(["torch"] if framework == "torch" else [framework]),
        data_contract_compatible=bool(compatibility.get("compatible", False)),
        remaining_blocker="architecture_factory_or_data_contract_not_integrated",
    )


def build_default_registry() -> ModelRegistry:
    """
    Registry of every model found in the audited old project.
    """

    sklearn_models = [
        ("ridge", "线性", False, "seconds"),
        ("bayesianridge", "线性", True, "seconds"),
        ("hgb", "树集成", False, "minutes"),
        ("svr", "核方法", False, "minutes"),
        ("rf", "树集成", False, "minutes"),
        ("mlp", "神经网络", False, "minutes"),
        ("elasticnet", "线性", False, "seconds"),
        ("pls", "化学计量学", False, "seconds"),
        ("knn", "非参数", False, "seconds"),
        ("gpr", "核方法/概率", True, "minutes"),
    ]

    deep_models = [
        (
            "transformer",
            "Transformer",
            "vendor/transformer/transformer.py", True,
        ),
        (
            "lstm",
            "循环神经网络",
            "vendor/legacy/lstm.py", True,
        ),
        (
            "dlinear",
            "线性/时序",
            "vendor/legacy/dlinear.py", True,
        ),
        (
            "gru",
            "循环神经网络",
            "vendor/packed_loader.py", True,
        ),
        (
            "patchtst",
            "Transformer",
            "vendor/tsl/models/PatchTST.py", False,
        ),
        (
            "itransformer",
            "Transformer",
            "vendor/tsl/models/iTransformer.py", False,
        ),
        (
            "timesnet",
            "时序卷积/周期",
            "vendor/tsl/models/TimesNet.py", False,
        ),
    ]

    specs: list[ModelSpec] = []

    # Persistence heuristic baseline.
    specs.append(
        ModelSpec(
            model_name="persistence",
            framework="heuristic",
            supported_tasks=["mass_flow_forecast", "steam_volume_forecast"],
            required_input_type="raw_window",
            required_features=30,
            feature_indices=list(range(30)),
            sequence_required=True,
            window_requirements={
                "steps": 20,
                "lookback_minutes": 5,
                "sampling_interval_seconds": 15,
            },
            horizon_capability={
                "steps": 40,
                "minutes": 10,
                "supported_steps": [40, 80],
            },
            supported_targets=list(_MASS_FLOW_TARGETS) + list(_VOLUME_TARGETS),
            training_available=True,
            inference_available=True,
            checkpoint_available=False,
            checkpoint_path=None,
            checkpoint_compatibility={
                "compatible": True,
                "checked": ["heuristic"],
                "mismatches": [],
                "note": (
                    "Heuristic baseline (last window value); "
                    "no checkpoint required."
                ),
            },
            supported_metrics=list(_METRICS),
            supports_uncertainty=False,
            compute_cost="trivial",
            status="benchmark_active",
            family="统计基线",
            source=OLD_LIBRARY,
            trainable=True,
            inference_supported=True,
            checkpoint_required=False,
            requires_torch=False,
            requires_cuda=False,
            cpu_supported=True,
            gpu_supported=False,
        )
    )

    # 10 unified sklearn benchmark models.
    for (
        name,
        family,
        uncertainty,
        cost,
    ) in sklearn_models:
        if name == "svr":
            # RBF SVR on ~17k x 600 flattened samples is
            # prohibitive without subsampling (the old project
            # subsampled it); mark it high-cost so autonomous
            # planning avoids it under the large-sample
            # contract.
            cost = "hours"

        specs.append(
            _sklearn_spec(
                name=name,
                family=family,
                uncertainty=uncertainty,
                cost=cost,
                checkpoint_path=(
                    f"{OLD_LIBRARY}\\weights"
                    f"\\benchmark_sklearn\\{name}.joblib"
                ),
            )
        )

    # 7 unified deep benchmark models.
    for name, family, source_path, adapter_ready in deep_models:
        specs.append(
            _deep_spec(
                name=name,
                family=family,
                source_code_path=source_path,
                adapter_ready=adapter_ready,
            )
        )

    # Legacy models with incompatible contracts.
    specs.append(
        _legacy_spec(
            name="mtgnn",
            family="图神经网络",
            framework="torch",
            features=30,
            window={
                "steps": 48,
                "lookback_minutes": 12,
                "sampling_interval_seconds": 15,
            },
            horizon={
                "steps": 1,
                "minutes": 0.25,
                "supported_steps": [1],
            },
            targets=list(_MASS_FLOW_TARGETS),
            input_type="sequence_window_31col",
            checkpoint_path=(
                f"{OLD_LIBRARY}\\weights"
                f"\\legacy_models\\mtgnn\\model_boiler.pt"
            ),
            compatibility=_compatible_checkpoint(
                window_ok=False,
                horizon_ok=False,
                note=(
                    "MTGNN uses a 48-step window and 1-step "
                    "horizon; incompatible with the current "
                    "20/40 contract. Reuse architecture only, "
                    "retrain required."
                ),
            ),
            status="legacy_review_only",
            cost="minutes",
            source_code_path=(
                "vendor/legacy/mtgnn/net.py"
            ),
        )
    )

    specs.append(
        _legacy_spec(
            name="csdi",
            family="扩散概率",
            framework="torch",
            features=181,
            window={
                "steps": 20,
                "lookback_minutes": 5,
                "sampling_interval_seconds": 15,
            },
            horizon={
                "steps": 40,
                "minutes": 10,
                "supported_steps": [40],
            },
            targets=list(_MASS_FLOW_TARGETS),
            input_type="sequence_window_181col",
            checkpoint_path=(
                f"{OLD_LIBRARY}\\weights"
                f"\\legacy_models\\csdi\\model.pth"
            ),
            compatibility=_compatible_checkpoint(
                features_ok=False,
                note=(
                    "CSDI uses 181 features; incompatible "
                    "with the current 30-feature contract. "
                    "Reuse architecture only, retrain required."
                ),
            ),
            status="legacy_review_only",
            uncertainty=True,
            cost="hours",
            source_code_path=(
                "vendor/legacy/csdi/main_model.py"
            ),
        )
    )

    specs.append(
        _legacy_spec(
            name="tcn",
            family="时间卷积",
            framework="torch",
            features=30,
            window={
                "steps": 120,
                "lookback_minutes": 30,
                "sampling_interval_seconds": 15,
            },
            horizon={
                "steps": 40,
                "minutes": 10,
                "supported_steps": [40],
            },
            targets=list(_MASS_FLOW_TARGETS),
            input_type="sequence_window",
            checkpoint_path=(
                f"{OLD_LIBRARY}\\weights"
                f"\\legacy_models\\tcn\\best.pth"
            ),
            compatibility=_compatible_checkpoint(
                window_ok=False,
                note=(
                    "TCN uses a 120-step window; incompatible "
                    "with the current 20-step contract. Reuse "
                    "architecture only, retrain required."
                ),
            ),
            status="legacy_review_only",
            cost="minutes",
            source_code_path=(
                "vendor/legacy/tcn.py"
            ),
        )
    )

    # PSFA steam-volume models (h0 / h20 / h80).
    psfa_specs = [
        (
            "psfa_v0",
            "h0_当前蒸汽量(软测量)",
            0,
            0,
        ),
        (
            "psfa_v20",
            "h20_5分钟后蒸汽量",
            20,
            5,
        ),
        (
            "psfa_v80",
            "h80_20分钟后蒸汽量",
            80,
            20,
        ),
    ]

    for (
        model_id,
        folder,
        horizon_steps,
        horizon_minutes,
    ) in psfa_specs:
        specs.append(
            _legacy_spec(
                name=model_id,
                family="概率慢特征",
                framework="sklearn",
                tasks=["steam_volume_forecast"],
                features=31,
                window={
                    "steps": 20,
                    "lookback_minutes": 5,
                    "sampling_interval_seconds": 15,
                },
                horizon={
                    "steps": horizon_steps,
                    "minutes": horizon_minutes,
                    "supported_steps": [horizon_steps],
                },
                targets=list(_VOLUME_TARGETS),
                input_type="flattened_window_31col",
                checkpoint_path=(
                    f"{OLD_LIBRARY}\\weights"
                    f"\\psfa\\{folder}\\ridge_ridge.joblib"
                ),
                compatibility=_compatible_checkpoint(
                    target_ok=False,
                    features_ok=False,
                    horizon_ok=(
                        horizon_steps in (0, 20, 80)
                    ),
                    note=(
                        "PSFA predicts volumetric flow V from "
                        "a 181-derived 31-column feature "
                        "scheme; incompatible with the current "
                        "M-target 30-feature contract. Reuse "
                        "architecture only, retrain required."
                    ),
                ),
                status="needs_validation",
                cost="seconds",
                source_code_path=None,
                trainable=False,
                inference_supported=False,
            )
        )

    # Registry-catalog legacy mass-flow models (DCS 7/9).
    legacy_mass_flow = [
        ("legacy_tcn_1min", "pred1min", "TCN", 4, 1, 40),
        ("legacy_tcn_2min", "pred2min", "TCN", 8, 2, 40),
        (
            "legacy_transformer_5min",
            "pred5min",
            "Transformer",
            20,
            5,
            40,
        ),
        (
            "legacy_transformer_10min",
            "pred10min",
            "Transformer",
            40,
            10,
            80,
        ),
    ]

    for (
        model_id,
        folder,
        arch,
        horizon_steps,
        horizon_minutes,
        lookback_steps,
    ) in legacy_mass_flow:
        specs.append(
            _legacy_spec(
                name=model_id,
                family=arch,
                framework="torch",
                features=2,
                window={
                    "steps": lookback_steps,
                    "lookback_minutes": (
                        lookback_steps * 15 // 60
                    ),
                    "sampling_interval_seconds": 15,
                },
                horizon={
                    "steps": horizon_steps,
                    "minutes": horizon_minutes,
                    "supported_steps": [horizon_steps],
                },
                targets=list(_MASS_FLOW_TARGETS),
                input_type="sequence_window_2col",
                checkpoint_path=(
                    r"D:\BoilerMindTeamTest\_bm_sync_tmp"
                    r"\model_registry\legacy_mass_flow_best_models"
                    f"\\{folder}\\model.pth"
                ),
                compatibility=_compatible_checkpoint(
                    features_ok=False,
                    window_ok=(
                        lookback_steps == 20
                    ),
                    note=(
                        f"{arch} trained on DCS 7/9 only "
                        f"({lookback_steps}-step lookback); "
                        "incompatible with the current "
                        "30-feature 20-step contract. Reuse "
                        "architecture only, retrain required."
                    ),
                ),
                status="legacy_review_only",
                cost="minutes",
                source_code_path=(
                    "vendor/legacy/tcn.py"
                    if arch == "TCN"
                    else (
                        "vendor/transformer/"
                        "transformer.py"
                    )
                ),
            )
        )

    # Direct-volume 20m candidates (XGBoost / LSTM).
    direct_volume = [
        (
            "candidate_xgboost_direct_volume_20m",
            "xgboost",
            "four independent XGBRegressor heads",
        ),
        (
            "candidate_lstm_direct_volume_20m",
            "torch",
            "64-hidden-unit causal LSTM, four heads",
        ),
    ]

    for (
        model_id,
        framework,
        architecture,
    ) in direct_volume:
        specs.append(
            _legacy_spec(
                name=model_id,
                family="直接体积流量",
                framework=framework,
                tasks=["steam_volume_forecast"],
                features=180,
                window={
                    "steps": 80,
                    "lookback_minutes": 20,
                    "sampling_interval_seconds": 15,
                },
                horizon={
                    "steps": 40,
                    "minutes": 10,
                    "supported_steps": [4, 8, 20, 40],
                },
                targets=list(_VOLUME_TARGETS),
                input_type="flattened_window_180col",
                checkpoint_path=None,
                compatibility=_compatible_checkpoint(
                    target_ok=False,
                    features_ok=False,
                    window_ok=False,
                    note=(
                        f"{architecture}; 80-step 180-feature "
                        "direct-IF97 volume contract, "
                        "review_only. Incompatible with the "
                        "current M-target contract."
                    ),
                ),
                status="needs_validation",
                cost="minutes",
                source_code_path=None,
                trainable=False,
                inference_supported=False,
            )
        )

    for spec in specs:
        tags = set(spec.capability_tags)
        if spec.sequence_required or spec.window_requirements.get("steps", 1) > 1:
            tags.add("temporal_dependency")
        lowered_name = spec.model_name.casefold()
        if spec.framework == "torch" or any(
            token in lowered_name
            for token in ("tcn", "transformer", "lstm", "gru", "timesnet")
        ):
            tags.add("sequence_model")
        if lowered_name.startswith("psfa") or spec.family == "概率慢特征":
            tags.add("physics_constraint")
        if lowered_name in {"elasticnet", "pls", "rf", "hgb"}:
            tags.add("feature_selection")
        if "optimization_surrogate" in spec.task_list:
            tags.add("optimization_surrogate")
        if spec.supports_uncertainty:
            tags.add("uncertainty_estimation")
        spec.capability_tags = sorted(tags)

    registry = ModelRegistry()
    registry.register_many(specs)

    return registry
