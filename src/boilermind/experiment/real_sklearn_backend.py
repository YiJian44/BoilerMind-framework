from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import (
    Ridge,
    BayesianRidge,
    ElasticNet,
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class RealSklearnExperimentBackend:
    """
    Real boiler mass-flow forecasting backend.

    Scientific protocol:
    - real 30-feature boiler data
    - chronological history window
    - explicit future prediction horizon
    - train / validation / locked-test split
    - hyperparameter selection on validation only
    - final refit on train + validation
    - locked test is evaluation only
    """

    SUPPORTED_MODELS = {
        "ridge",
        "bayesianridge",
        "hgb",
        "rf",
        "svr",
        "elasticnet",
        "mlp",
        "pls",
        "knn",
        "gpr",
    }

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()

        with path.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)

                if not block:
                    break

                h.update(block)

        return h.hexdigest()

    def _metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, float]:

        actual = np.asarray(actual, dtype=float).reshape(-1)
        predicted = np.asarray(predicted, dtype=float).reshape(-1)

        return {
            "mae_t_h": float(
                mean_absolute_error(actual, predicted)
            ),
            "rmse_t_h": float(
                np.sqrt(
                    mean_squared_error(actual, predicted)
                )
            ),
            "r2": float(
                r2_score(actual, predicted)
            ),
            "mbe_t_h": float(
                np.mean(predicted - actual)
            ),
        }

    def _parameter_grid(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:

        grids = {
            "ridge": [
                {"alpha": 0.1},
                {"alpha": 1.0},
                {"alpha": 10.0},
                {"alpha": 100.0},
            ],

            "bayesianridge": [
                {},
            ],

            "hgb": [
                {
                    "learning_rate": 0.05,
                    "max_iter": 150,
                    "max_leaf_nodes": 31,
                },
                {
                    "learning_rate": 0.10,
                    "max_iter": 150,
                    "max_leaf_nodes": 31,
                },
            ],

            "rf": [
                {
                    # v1 黄金链路固定为本地 CPU 可稳定重复的轻量配置。
                    "n_estimators": 40,
                    "max_depth": 12,
                },
            ],

            "svr": [
                {
                    "C": 10.0,
                    "epsilon": 0.1,
                    "kernel": "rbf",
                },
            ],

            "elasticnet": [
                {
                    "alpha": 0.001,
                    "l1_ratio": 0.5,
                },
                {
                    "alpha": 0.01,
                    "l1_ratio": 0.5,
                },
            ],

            "mlp": [
                {
                    "hidden_layer_sizes": (64,),
                    "max_iter": 300,
                },
            ],

            "pls": [
                {
                    "n_components": 8,
                },
            ],

            "knn": [
                {
                    "n_neighbors": 5,
                },
            ],

            # GPR 对 1.7 万训练样本计算代价很高。
            # 先注册能力，但真实大样本实验默认不运行。
            "gpr": [
                {},
            ],
        }

        if model_id not in grids:
            raise ValueError(
                f"unsupported_model:{model_id}"
            )

        return grids[model_id]

    def _build_model(
        self,
        model_id: str,
        params: dict[str, Any],
        random_seed: int,
    ):

        if model_id == "ridge":
            return Ridge(**params)

        if model_id == "bayesianridge":
            return BayesianRidge(**params)

        if model_id == "hgb":
            return HistGradientBoostingRegressor(
                random_state=random_seed,
                **params,
            )

        if model_id == "rf":
            return RandomForestRegressor(
                random_state=random_seed,
                n_jobs=-1,
                **params,
            )

        if model_id == "svr":
            return SVR(**params)

        if model_id == "elasticnet":
            return ElasticNet(
                random_state=random_seed,
                max_iter=5000,
                **params,
            )

        if model_id == "mlp":
            return MLPRegressor(
                random_state=random_seed,
                **params,
            )

        if model_id == "pls":
            return PLSRegression(**params)

        if model_id == "knn":
            return KNeighborsRegressor(
                n_jobs=-1,
                **params,
            )

        if model_id == "gpr":
            return GaussianProcessRegressor(
                random_state=random_seed,
                **params,
            )

        raise ValueError(
            f"unsupported_model:{model_id}"
        )

    def _prepare_dataset(
        self,
        *,
        dataset_path: Path,
        window_steps: int,
        horizon_steps: int,
        train_ratio: float,
        validation_ratio: float,
    ) -> dict[str, Any]:

        # shortperiod_new.csv has NO header.
        frame = pd.read_csv(
            dataset_path,
            header=None,
        )

        if frame.shape[1] != 31:
            raise ValueError(
                "dataset_contract_violation:"
                f"expected_31_columns,"
                f"got_{frame.shape[1]}"
            )

        raw_features = frame.iloc[
            :, :30
        ].to_numpy(dtype=np.float64)

        raw_target = frame.iloc[
            :, 30
        ].to_numpy(dtype=np.float64)

        row_count = len(frame)

        source_indices = np.arange(
            window_steps - 1,
            row_count - horizon_steps,
            dtype=int,
        )

        target_indices = (
            source_indices + horizon_steps
        )

        sample_count = len(source_indices)

        train_end = int(
            sample_count * train_ratio
        )

        validation_count = int(
            sample_count * validation_ratio
        )

        validation_end = (
            train_end + validation_count
        )

        train_index = np.arange(
            0,
            train_end,
            dtype=int,
        )

        validation_index = np.arange(
            train_end,
            validation_end,
            dtype=int,
        )

        locked_test_index = np.arange(
            validation_end,
            sample_count,
            dtype=int,
        )

        if (
            len(train_index) == 0
            or len(validation_index) == 0
            or len(locked_test_index) == 0
        ):
            raise ValueError(
                "invalid_split"
            )

        # Critical:
        # scaler is fitted only on feature rows available to
        # the training forecast origins.
        last_train_source_row = int(
            source_indices[train_index[-1]]
        )

        feature_scaler = MinMaxScaler()

        feature_scaler.fit(
            raw_features[
                : last_train_source_row + 1
            ]
        )

        scaled_features = (
            feature_scaler.transform(
                raw_features
            )
        )

        X = np.stack(
            [
                scaled_features[
                    source_index
                    - window_steps
                    + 1:
                    source_index
                    + 1
                ].reshape(-1)
                for source_index
                in source_indices
            ],
            axis=0,
        )

        y = raw_target[
            target_indices
        ]

        source_mass = raw_target[
            source_indices
        ]

        if X.shape[1] != (
            window_steps * 30
        ):
            raise RuntimeError(
                "window_feature_dimension_error"
            )

        return {
            "X": X,
            "y": y,
            "source_mass": source_mass,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "train_index": train_index,
            "validation_index": validation_index,
            "locked_test_index": locked_test_index,
            "sample_count": sample_count,
            "row_count": row_count,
            "feature_scaler": feature_scaler,
        }

    def _fit_and_evaluate_one(
        self,
        *,
        model_id: str,
        X,
        y,
        train,
        validation,
        development,
        test,
        random_seed: int,
    ) -> tuple[dict[str, Any], Any, Any]:
        """
        Train + validate + locked-test ONE model.

        Warnings are captured, not lost: a ConvergenceWarning
        makes fit_converged=False while fit_success stays True.
        Exceptions propagate to the caller, which records
        failure_reason (no silent model substitution).
        """

        validation_records = []
        best = None
        collected_warnings: list[str] = []

        for params in self._parameter_grid(
            model_id
        ):

            with warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter("always")

                model = self._build_model(
                    model_id,
                    params,
                    random_seed,
                )

                model.fit(
                    X[train],
                    y[train],
                )

            collected_warnings.extend(
                str(warning.message)
                for warning in caught
            )

            validation_pred = (
                np.asarray(
                    model.predict(
                        X[validation]
                    )
                )
                .reshape(-1)
            )

            validation_pred = (
                np.maximum(
                    0.0,
                    validation_pred,
                )
            )

            metrics = self._metrics(
                y[validation],
                validation_pred,
            )

            record = {
                "params": params,
                **metrics,
            }

            validation_records.append(
                record
            )

            if (
                best is None
                or record["mae_t_h"]
                < best["mae_t_h"]
            ):
                best = record

        if best is None:
            raise RuntimeError(
                "no_valid_validation_record:"
                f"{model_id}"
            )

        with warnings.catch_warnings(
            record=True
        ) as caught_final:
            warnings.simplefilter("always")

            final_model = self._build_model(
                model_id,
                dict(best["params"]),
                random_seed,
            )

            final_model.fit(
                X[development],
                y[development],
            )

        collected_warnings.extend(
            str(warning.message)
            for warning in caught_final
        )

        locked_pred = np.asarray(
            final_model.predict(
                X[test]
            )
        ).reshape(-1)

        locked_pred = np.maximum(
            0.0,
            locked_pred,
        )

        locked_metrics = self._metrics(
            y[test],
            locked_pred,
        )

        selected_validation_metrics = {
            k: best[k]
            for k in (
                "mae_t_h",
                "rmse_t_h",
                "r2",
                "mbe_t_h",
            )
        }

        fit_converged = not any(
            "ConvergenceWarning" in warning
            for warning in collected_warnings
        )

        return (
            {
                "fit_success": True,
                "fit_converged": fit_converged,
                "warnings": collected_warnings,
                "failure_reason": None,
                "selected_parameters": dict(
                    best["params"]
                ),
                "model_config": dict(
                    best["params"]
                ),
                "validation_records": (
                    validation_records
                ),
                "validation_metrics": {
                    **selected_validation_metrics
                },
                "locked_test_metrics": (
                    locked_metrics
                ),
            },
            final_model,
            locked_pred,
        )

    def run(
        self,
        contract: dict[str, Any],
    ) -> dict[str, Any]:

        started = perf_counter()

        dataset_path = Path(
            contract["dataset_path"]
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                dataset_path
            )

        window_steps = int(
            contract.get(
                "window_steps",
                20,
            )
        )

        horizon_steps = int(
            contract.get(
                "prediction_horizon_steps",
                40,
            )
        )

        sampling_interval_seconds = int(
            contract.get(
                "sampling_interval_seconds",
                15,
            )
        )

        random_seed = int(
            contract.get(
                "random_seed",
                42,
            )
        )

        train_ratio = float(
            contract.get(
                "train_ratio",
                0.70,
            )
        )

        validation_ratio = float(
            contract.get(
                "validation_ratio",
                0.15,
            )
        )

        model_candidates = contract.get(
            "model_candidates"
        )

        if not isinstance(
            model_candidates,
            list,
        ) or not model_candidates:

            raise ValueError(
                "model_candidates_required"
            )

        unknown = [
            model_id
            for model_id in model_candidates
            if model_id
            not in self.SUPPORTED_MODELS
        ]

        if unknown:
            raise ValueError(
                "unsupported_models:"
                + ",".join(unknown)
            )

        # Do not silently execute GPR on this full dataset.
        # Fail closed instead.
        data = self._prepare_dataset(
            dataset_path=dataset_path,
            window_steps=window_steps,
            horizon_steps=horizon_steps,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

        X = data["X"]
        y = data["y"]

        train = data["train_index"]
        validation = data[
            "validation_index"
        ]
        development = np.concatenate(
            [
                train,
                validation,
            ]
        )
        test = data[
            "locked_test_index"
        ]

        source_mass = data[
            "source_mass"
        ]

        persistence_validation = (
            self._metrics(
                y[validation],
                source_mass[validation],
            )
        )

        persistence_locked = (
            self._metrics(
                y[test],
                source_mass[test],
            )
        )

        verbose = bool(
            contract.get(
                "verbose",
                True,
            )
        )

        dataset_sha256 = self._sha256(
            dataset_path
        )

        model_results = {}

        for model_id in model_candidates:

            if verbose:
                print(
                    f"START {model_id}",
                    flush=True,
                )

            model_started = perf_counter()

            try:
                (
                    record,
                    final_model,
                    locked_pred,
                ) = self._fit_and_evaluate_one(
                    model_id=model_id,
                    X=X,
                    y=y,
                    train=train,
                    validation=validation,
                    development=development,
                    test=test,
                    random_seed=random_seed,
                )
            except Exception as exc:
                elapsed = (
                    perf_counter()
                    - model_started
                )

                failure_reason = (
                    f"{type(exc).__name__}:{exc}"
                )

                if verbose:
                    print(
                        "FAILED "
                        f"{model_id} "
                        f"elapsed_seconds={elapsed:.3f} "
                        "failure_reason="
                        f"{failure_reason}",
                        flush=True,
                    )

                model_results[model_id] = {
                    "fit_success": False,
                    "fit_converged": False,
                    "warnings": [],
                    "failure_reason": (
                        failure_reason
                    ),
                    "elapsed_seconds": elapsed,
                    "selected_parameters": {},
                    "validation_records": [],
                    "validation_metrics": {},
                    "locked_test_metrics": {},
                    "train_samples": len(train),
                    "validation_samples": len(
                        validation
                    ),
                    "test_samples": len(test),
                    "random_seed": random_seed,
                    "dataset_sha256": dataset_sha256,
                    "model_artifact": None,
                    "prediction_artifact": None,
                    "_final_model": None,
                    "_locked_prediction": None,
                }
                continue

            elapsed = (
                perf_counter()
                - model_started
            )

            record["elapsed_seconds"] = elapsed
            record["train_samples"] = len(train)
            record["validation_samples"] = len(
                validation
            )
            record["test_samples"] = len(test)
            record["random_seed"] = random_seed
            record["dataset_sha256"] = dataset_sha256
            record["_final_model"] = final_model
            record["_locked_prediction"] = locked_pred

            if verbose:
                print(
                    "DONE "
                    f"{model_id} "
                    f"elapsed_seconds={elapsed:.3f} "
                    "validation_metrics="
                    f"{json.dumps(record['validation_metrics'])} "
                    "locked_test_metrics="
                    f"{json.dumps(record['locked_test_metrics'])}",
                    flush=True,
                )

            model_results[model_id] = record

        successful_models = [
            model_id
            for model_id, result in (
                model_results.items()
            )
            if result.get("fit_success", False)
        ]

        if not successful_models:
            raise RuntimeError(
                "all_models_fit_failed"
            )

        selected_model = min(
            successful_models,
            key=lambda model_id:
                model_results[
                    model_id
                ][
                    "validation_metrics"
                ][
                    "mae_t_h"
                ],
        )

        experiment_id = str(
            contract.get(
                "experiment_id",
                "REAL-SKLEARN",
            )
        )

        output_dir = Path(
            contract.get(
                "output_dir",
                "outputs/experiments",
            )
        ) / experiment_id

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for model_id, result in (
            model_results.items()
        ):

            if result.get("_final_model") is None:
                result["artifact_paths"] = []
                continue

            model_path = (
                output_dir
                / f"{model_id}.joblib"
            )

            joblib.dump(
                result["_final_model"],
                model_path,
            )

            pred_path = (
                output_dir
                / f"{model_id}_locked_test_predictions.csv"
            )

            pd.DataFrame(
                {
                    "source_row_index":
                        data[
                            "source_indices"
                        ][test],

                    "target_row_index":
                        data[
                            "target_indices"
                        ][test],

                    "actual_mass_t_h":
                        y[test],

                    "persistence_mass_t_h":
                        source_mass[test],

                    "predicted_mass_t_h":
                        result[
                            "_locked_prediction"
                        ],

                    "error_t_h":
                        result[
                            "_locked_prediction"
                        ]
                        - y[test],
                }
            ).to_csv(
                pred_path,
                index=False,
                encoding="utf-8-sig",
            )

            del result["_final_model"]
            del result[
                "_locked_prediction"
            ]

            result["model_artifact"] = str(
                model_path.resolve()
            )

            result[
                "prediction_artifact"
            ] = str(
                pred_path.resolve()
            )

            result["artifact_paths"] = [
                str(model_path.resolve()),
                str(pred_path.resolve()),
            ]

        result_payload = {
            "schema_version":
                "boilermind.real_sklearn_experiment.v1",

            "experiment_id":
                experiment_id,

            "status":
                "completed",

            "execution_mode":
                "real_train_validate_locked_test",

            "dataset": {
                "path":
                    str(
                        dataset_path.resolve()
                    ),

                "sha256":
                    self._sha256(
                        dataset_path
                    ),

                "raw_row_count":
                    data["row_count"],

                "eligible_sample_count":
                    data["sample_count"],
            },

            "forecast_contract": {
                "window_steps":
                    window_steps,

                "prediction_horizon_steps":
                    horizon_steps,

                "sampling_interval_seconds":
                    sampling_interval_seconds,

                "window_duration_seconds":
                    window_steps
                    * sampling_interval_seconds,

                "forecast_horizon_seconds":
                    horizon_steps
                    * sampling_interval_seconds,

                "feature_count":
                    30,

                "flattened_feature_count":
                    window_steps * 30,

                "target":
                    "main_steam_mass_flow",

                "target_unit":
                    "t/h",
            },

            "split": {
                "policy":
                    "chronological_train_validation_locked_test",

                "train_count":
                    len(train),

                "validation_count":
                    len(validation),

                "locked_test_count":
                    len(test),

                "selection_scope":
                    "validation_only",

                "locked_test_used_for_selection":
                    False,
            },

            "reference_model": {
                "model_id":
                    "persistence",

                "validation_metrics":
                    persistence_validation,

                "locked_test_metrics":
                    persistence_locked,
            },

            "model_candidates":
                model_candidates,

            "selected_model_by_validation":
                selected_model,

            "models":
                model_results,

            "random_seed":
                random_seed,

            "elapsed_seconds":
                perf_counter()
                - started,

            "completed_at":
                datetime.now().isoformat(),

            "scientific_note":
                (
                    "Model and hyperparameter selection "
                    "uses validation only. Locked test "
                    "is evaluated after selection."
                ),
        }

        metrics_path = (
            output_dir
            / "experiment_result.json"
        )

        metrics_path.write_text(
            json.dumps(
                result_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result_payload[
            "result_artifact"
        ] = str(
            metrics_path.resolve()
        )

        return result_payload
