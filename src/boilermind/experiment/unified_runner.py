from __future__ import annotations

import hashlib
import importlib.metadata
import json
import csv
import platform
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from boilermind.audit.execution_trace import ExperimentExecutionTrace
from boilermind.core.contracts.experiment import (
    ExperimentContract,
    ExperimentResult,
    ModelExperimentRecord,
)
from boilermind.core.enums import ExperimentStatus
from boilermind.experiment.capability_registry import ExperimentCapabilityRegistry
from boilermind.experiment.execution_policy import ExecutionPolicy
from boilermind.experiment.metric_normalizer import (
    normalize_metrics,
    numeric_normalized_metrics,
)
from boilermind.experiment.time_series_data import DatasetBuilder, TimeSeriesDataContract
from boilermind.models.catalog import build_default_registry
from boilermind.models.execution_environment import ExecutionEnvironment


class ExperimentExecutionError(RuntimeError):
    def __init__(self, message: str, result: ExperimentResult | None = None):
        super().__init__(message)
        self.result = result


class BackendResolver:
    """Framework-driven routing only; model names never appear here."""

    BINDINGS = {"sklearn": "sklearn", "torch": "torch", "heuristic": "reference"}

    def resolve(self, spec) -> str:
        try:
            return self.BINDINGS[spec.framework]
        except KeyError as exc:
            raise ValueError(f"backend_unavailable_for_framework:{spec.framework}") from exc


class UnifiedExperimentRunner:
    is_test_only = False

    MASS_FLOW_TARGET = "main_steam_mass_flow"
    DIRECT_VOLUME_TARGET = "steam_volumetric_flow"
    DIRECT_VOLUME_FEATURES = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 15, 18, 26, 30, 31, 33, 34,
        100, 101, 102, 103, 108, 109, 111, 112, 117, 118, 125, 126, 168, 175,
    )
    # horizon=0 表示"当前时刻"软测量（目标=窗末）；40/80 为前视预测。
    DIRECT_VOLUME_HORIZONS = (0, 40, 80)
    DIRECT_VOLUME_WINDOW_STEPS = (10, 20, 40, 80)
    # 与 model_library.json 的 14 个基础族对齐（persistence 为 reference 基线除外）。
    DIRECT_VOLUME_MODELS = frozenset(
        {
            "ridge", "bayesianridge", "elasticnet", "pls", "svr", "rf",
            "mlp", "knn", "hgb", "transformer", "lstm", "gru", "dlinear",
        }
    )
    DIRECT_VOLUME_DATASET = (
        Path(__file__).resolve().parents[3]
        / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"
    )

    def __init__(self, *, capability_registry=None, model_registry=None, dataset_builder=None,
                 backend_resolver=None, environment=None, output_dir=None):
        self.capability = capability_registry or ExperimentCapabilityRegistry()
        self.model_registry = model_registry or build_default_registry()
        self.dataset_builder = dataset_builder or DatasetBuilder()
        self.backend_resolver = backend_resolver or BackendResolver()
        self.environment = environment or ExecutionEnvironment.detect()
        self.output_dir = Path(output_dir or "outputs/experiments")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _prepare(self, contract: ExperimentContract, feature_columns=None):
        target = contract.target_variable.strip().lower()
        if target == self.DIRECT_VOLUME_TARGET:
            return self._prepare_direct_volume(contract, feature_columns)
        if target != self.MASS_FLOW_TARGET:
            raise ValueError(f"execution_target_not_supported:{contract.target_variable}")

        path = Path(self.capability.dataset_path)
        actual_hash = self._sha256(path)
        if contract.dataset_hash not in {"real", actual_hash}:
            raise ValueError("contract_dataset_hash_mismatch")
        data = self.dataset_builder.build_from_csv(path, TimeSeriesDataContract(
            feature_columns=tuple(range(30) if feature_columns is None else feature_columns), target_columns=(30,),
            sampling_interval_seconds=int(contract.sampling_interval_seconds or self.capability.sampling_interval_seconds),
            window_steps=int(contract.window_steps or self.capability.window_steps),
            prediction_horizon=int(contract.prediction_horizon_steps or self.capability.prediction_horizon_steps),
            train_ratio=self.capability.train_ratio, validation_ratio=self.capability.validation_ratio,
        ))
        return path, actual_hash, data

    def _prepare_direct_volume(self, contract: ExperimentContract, feature_columns=None):
        if feature_columns is not None:
            raise ValueError("direct_volume_feature_intervention_not_supported")
        horizon = int(contract.prediction_horizon_steps or 0)
        if horizon not in self.DIRECT_VOLUME_HORIZONS:
            raise ValueError(f"direct_volume_horizon_not_supported:{horizon}")
        window_steps = int(contract.window_steps or 0)
        if window_steps not in self.DIRECT_VOLUME_WINDOW_STEPS:
            raise ValueError(
                f"direct_volume_window_steps_not_supported:{window_steps}"
            )
        if int(contract.sampling_interval_seconds or 0) != 15:
            raise ValueError("direct_volume_sampling_interval_must_equal_15")

        path = Path(
            contract.execution_requirements.get("dataset_path")
            or self.DIRECT_VOLUME_DATASET
        )
        if not path.is_file():
            raise FileNotFoundError(f"direct_volume_dataset_not_found:{path}")
        actual_hash = self._sha256(path)
        if contract.dataset_hash not in {"real", actual_hash}:
            raise ValueError("contract_dataset_hash_mismatch")

        import pandas as pd
        frame = pd.read_csv(path, header=0)
        required = {str(column) for column in self.DIRECT_VOLUME_FEATURES}
        required.update({"1", "9", "16"})
        missing = sorted(required - set(map(str, frame.columns)))
        if missing:
            raise ValueError("direct_volume_columns_missing:" + ",".join(missing))
        pressure = frame["1"].to_numpy(dtype=float)
        temperature = frame["9"].to_numpy(dtype=float)
        mass_flow = frame["16"].to_numpy(dtype=float)
        frame = frame.copy()
        frame["__steam_volumetric_flow__"] = (
            mass_flow * (1000.0 / 3600.0) * 0.461526
            * (temperature + 273.15) / (pressure * 1000.0)
        )
        data = self.dataset_builder.build(
            frame,
            TimeSeriesDataContract(
                feature_columns=tuple(str(column) for column in self.DIRECT_VOLUME_FEATURES),
                target_columns=("__steam_volumetric_flow__",),
                sampling_interval_seconds=15,
                window_steps=window_steps,
                prediction_horizon=horizon,
                train_ratio=0.70,
                validation_ratio=0.10,
            ),
        )
        return path, actual_hash, data

    @staticmethod
    def _shape_for(spec, X):
        if spec.required_input_type == "flattened_window":
            return X.reshape(len(X), -1)
        if spec.required_input_type in {"sequence_window", "raw_window"}:
            return X
        raise ValueError(f"data_contract_incompatible:{spec.model_name}:{spec.required_input_type}")

    def _record_failure(self, name, exc, elapsed, contract, digest, data):
        return ModelExperimentRecord(model_name=name, fit_success=False, fit_converged=False,
            failure_reason=f"{type(exc).__name__}:{exc}", runtime_seconds=elapsed,
            train_samples=len(data.X_train), validation_samples=len(data.X_validation),
            test_samples=len(data.X_locked_test), random_seed=contract.random_seed,
            dataset_sha256=digest, artifact_provenance={"training_mode": "failed"})

    def _execute_model(self, name, contract, policy, digest, data, models_dir, artifact_stem=None):
        spec = self.model_registry.get(name)
        backend = self.backend_resolver.resolve(spec)
        target = contract.target_variable.strip().lower()
        if (
            target == self.DIRECT_VOLUME_TARGET
            and name not in self.DIRECT_VOLUME_MODELS
            and backend != "reference"
        ):
            raise RuntimeError(f"direct_volume_model_not_verified:{name}")
        if (
            target != self.DIRECT_VOLUME_TARGET
            and name not in self.capability.available_models()
            and backend != "reference"
        ):
            raise RuntimeError(f"planned_model_not_currently_executable:{name}")
        reuse = name in contract.reuse_checkpoint_models
        device = None
        if backend == "torch":
            device = "cuda" if self.environment.cuda_available and "cuda" in policy.allowed_devices else "cpu"
            policy.validate_device(device)
        adapter_kwargs = {"random_seed": contract.random_seed} if backend in {"sklearn", "torch"} else {}
        if backend == "torch":
            adapter_kwargs.update({"reuse_checkpoint": reuse, "device": device})
        adapter = self.model_registry.build_adapter(name, **adapter_kwargs)
        X_train = self._shape_for(spec, data.X_train)
        X_validation = self._shape_for(spec, data.X_validation)
        X_test = self._shape_for(spec, data.X_locked_test)
        if name == "gpr" and len(X_train) > 5000:
            raise RuntimeError(
                f"RESOURCE_SAFETY_BLOCKED:exact_gpr_train_samples:{len(X_train)}:limit:5000"
            )
        kwargs: dict[str, Any] = {}
        if backend == "torch":
            kwargs["validation_data"] = (X_validation, data.y_validation)
            if policy.max_epochs is not None:
                kwargs["epochs"] = policy.max_epochs
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter.fit(X_train, data.y_train.reshape(-1), **kwargs)
        elapsed = time.perf_counter() - fit_started
        captured_warnings = [f"{type(item.message).__name__}: {item.message}" for item in caught]
        captured_warnings.extend(getattr(adapter, "warning_messages", []))
        if policy.max_runtime_per_model is not None and elapsed > policy.max_runtime_per_model:
            raise TimeoutError(f"max_runtime_per_model_exceeded:{elapsed:.6f}")
        validation_predictions = np.asarray(adapter.predict(X_validation)).reshape(-1)
        locked_predictions = np.asarray(adapter.predict(X_test)).reshape(-1)
        validation = self._canonical_metrics(
            adapter.evaluate(data.y_validation, validation_predictions)
        )
        locked = self._canonical_metrics(
            adapter.evaluate(data.y_locked_test, locked_predictions)
        )
        artifact_paths = []
        checkpoint_provenance = {"training_mode": "reuse_checkpoint" if reuse else "train_from_source",
            "checkpoint_path": spec.checkpoint_path if reuse else None,
            "checkpoint_compatible": spec.checkpoint_compatibility.get("compatible") if reuse else None}
        if backend == "torch" and not reuse:
            checkpoint = models_dir / f"{artifact_stem or name}.pth"
            adapter._load_torch().save(adapter.model.state_dict(), checkpoint)
            artifact_paths.append(str(checkpoint.resolve()))
            checkpoint_provenance["generated_checkpoint"] = str(checkpoint.resolve())
            scaler_path = Path(str(checkpoint) + ".target_scaler.json")
            scaler_payload = {
                "method": getattr(adapter, "target_scaling_method", None),
                "fit_scope": "train_only",
                "mean": np.asarray(adapter.target_mean_).reshape(-1).tolist(),
                "scale": np.asarray(adapter.target_scale_).reshape(-1).tolist(),
                "prediction_inverse_transformed": True,
                "metrics_computed_in_original_unit": True,
            }
            scaler_path.write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")
            artifact_paths.append(str(scaler_path.resolve()))
            checkpoint_provenance.update(scaler_payload)
        elif backend == "sklearn":
            import joblib
            model_path = models_dir / f"{artifact_stem or name}.joblib"
            joblib.dump(adapter.estimator, model_path)
            artifact_paths.append(str(model_path.resolve()))
            checkpoint_provenance["generated_model"] = str(model_path.resolve())
        predictions_dir = models_dir.parent / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        protocol_sha256 = hashlib.sha256(
            contract.model_dump_json().encode("utf-8")
        ).hexdigest()
        for split_name, truth, prediction, source_indices, target_indices in (
            ("validation", data.y_validation, validation_predictions,
             data.source_indices["validation"], data.target_indices["validation"]),
            ("locked_test", data.y_locked_test, locked_predictions,
             data.source_indices["locked_test"], data.target_indices["locked_test"]),
        ):
            prediction_path = predictions_dir / f"{artifact_stem or name}_{split_name}_predictions.csv"
            with prediction_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("experiment_id", "model_id", "seed", "split", "source_index",
                                 "target_index", "y_true", "y_pred", "error", "abs_error",
                                 "dataset_sha256", "protocol_sha256"))
                for source_index, target_index, y_true, y_pred in zip(
                    source_indices, target_indices, np.asarray(truth).reshape(-1), prediction
                ):
                    error = float(y_pred - y_true)
                    writer.writerow((contract.experiment_id, name, contract.random_seed, split_name,
                                     int(source_index), int(target_index), float(y_true), float(y_pred),
                                     error, abs(error), digest, protocol_sha256))
            artifact_paths.append(str(prediction_path.resolve()))
        return ModelExperimentRecord(model_name=name, fit_success=True,
            fit_converged=not any("ConvergenceWarning" in item for item in captured_warnings),
            warnings=captured_warnings, runtime_seconds=getattr(adapter, "runtime_seconds", None) or elapsed,
            model_configuration=dict(getattr(adapter, "config", {}) or getattr(adapter, "params", {}) or {}),
            validation_metrics=validation, locked_test_metrics=locked,
            train_samples=len(data.X_train), validation_samples=len(data.X_validation), test_samples=len(data.X_locked_test),
            random_seed=contract.random_seed, dataset_sha256=digest, artifact_paths=artifact_paths,
            device=str(getattr(adapter, "device", "cpu")), epochs_completed=getattr(adapter, "epochs_completed", None),
            best_epoch=getattr(adapter, "best_epoch", None), training_loss=getattr(adapter, "training_loss", None),
            validation_loss=getattr(adapter, "validation_loss", None), artifact_provenance=checkpoint_provenance)

    @staticmethod
    def _canonical_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """保留底层单位键，同时提供合同使用的统一指标名。"""
        normalized = dict(metrics)
        normalized.update(numeric_normalized_metrics(metrics))
        return normalized

    @classmethod
    def _metric_value(cls, metrics: dict[str, float], metric: str) -> float:
        normalized = cls._canonical_metrics(metrics)
        if metric not in normalized:
            raise ValueError(f"required_metric_missing:{metric}")
        return float(normalized[metric])

    def _write_artifacts(self, contract, result, digest, started, experiment_dir):
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for child in ("models", "metrics", "logs"):
            (experiment_dir / child).mkdir(exist_ok=True)
        contract_path = experiment_dir / "contract.json"
        result_path = experiment_dir / "result.json"
        manifest_path = experiment_dir / "manifest.json"
        metrics_path = experiment_dir / "metrics" / "model_metrics.json"
        log_path = experiment_dir / "logs" / "execution.json"
        contract_path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
        manifest = {"schema_version": "boilermind.unified_experiment.v1",
            "prediction_schema_version": "boilermind.predictions.v1",
            "problem_id": contract.problem_id, "hypothesis_id": contract.hypothesis_id,
            "plan_id": contract.plan_id, "experiment_id": contract.experiment_id,
            "dataset_id": contract.dataset_id, "dataset_sha256": digest,
            "target_variable": contract.target_variable,
            "input_variables": list(contract.input_variables),
            "window_steps": contract.window_steps,
            "prediction_horizon_steps": contract.prediction_horizon_steps,
            "sampling_interval_seconds": contract.sampling_interval_seconds,
            "candidate_models": list(contract.candidate_models), "reference_models": list(contract.reference_models),
            "random_seed": contract.random_seed, "python_version": sys.version,
            "dependency_versions": {name: self._package_version(name) for name in ("boilermind-trusted", "numpy", "pandas", "scikit-learn", "torch")},
            "platform": platform.platform(), "environment": self.environment.to_dict(),
            "deterministic_configuration_recorded": True, "bit_level_reproducibility_claimed": False,
            "started_at": started.isoformat(), "completed_at": result.completed_at.isoformat() if result.completed_at else None}
        metrics_path.write_text(json.dumps({name: {"validation": record.validation_metrics,
            "locked_test": record.locked_test_metrics} for name, record in result.model_records.items()}, indent=2), encoding="utf-8")
        log_path.write_text(json.dumps({name: {"fit_success": record.fit_success, "warnings": record.warnings,
            "failure_reason": record.failure_reason, "runtime_seconds": record.runtime_seconds}
            for name, record in result.model_records.items()}, indent=2), encoding="utf-8")
        paths = [str(p.resolve()) for p in (manifest_path, contract_path, result_path, metrics_path, log_path)]
        result.artifacts.extend(paths)
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return paths

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def run(self, contract: ExperimentContract):
        if not isinstance(contract, ExperimentContract):
            raise TypeError("experiment_contract_required")
        frozen_models = tuple(contract.candidate_models)
        if len(frozen_models) != len(set(frozen_models)):
            raise ValueError("duplicate_candidate_models")
        if contract.locked_test_used_for_selection:
            raise ValueError("locked_test_used_for_selection")
        regime_operation = "regime_stratified_evaluation"
        regime_declared = regime_operation in set(contract.required_operations)
        regime_type = contract.experiment_type == regime_operation
        if regime_declared != regime_type:
            raise ValueError(
                "regime_operation_contract_mismatch:"
                f"operation={regime_declared}:experiment_type={regime_type}"
            )
        random.seed(contract.random_seed)
        np.random.seed(contract.random_seed)
        policy = ExecutionPolicy.from_contract(contract)
        started = datetime.now(timezone.utc)
        if contract.experiment_type in {"feature_ablation", "feature_intervention"}:
            return self._run_feature_intervention(contract, policy, started)
        if contract.experiment_type == "regime_stratified_evaluation":
            return self._run_regime_stratified(contract, policy, started)
        path, digest, data = self._prepare(contract)
        experiment_dir = self.output_dir / contract.experiment_id
        models_dir = experiment_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        records = {}
        for name in frozen_models:
            model_started = time.perf_counter()
            try:
                records[name] = self._execute_model(name, contract, policy, digest, data, models_dir)
            except Exception as exc:
                records[name] = self._record_failure(name, exc, time.perf_counter() - model_started, contract, digest, data)
        if tuple(records) != frozen_models:
            raise RuntimeError("runner_mutated_frozen_model_set")
        successful = [name for name, record in records.items() if record.fit_success]
        selected = min(
            successful,
            key=lambda name: self._metric_value(
                records[name].validation_metrics, "MAE"
            ),
        ) if successful else None
        persistence_adapter = self.model_registry.build_adapter("persistence")
        reference_metrics = self._canonical_metrics(
            persistence_adapter.evaluate(data.y_locked_test, data.y_source_locked_test)
        )
        candidates = {name: record.locked_test_metrics for name, record in records.items() if record.fit_success}
        candidates["persistence"] = reference_metrics
        raw_result_metrics = records[selected].locked_test_metrics if selected else {}
        result = ExperimentResult(experiment_id=contract.experiment_id, problem_id=contract.problem_id,
            hypothesis_id=contract.hypothesis_id, plan_id=contract.plan_id,
            status=ExperimentStatus.COMPLETED if successful else ExperimentStatus.FAILED,
            metrics=self._canonical_metrics(raw_result_metrics),
            raw_metrics=raw_result_metrics,
            normalized_metrics=normalize_metrics(raw_result_metrics),
            baseline_metrics=reference_metrics,
            candidate_locked_test_metrics=candidates, model_records=records,
            execution_notes=["UNIFIED_EXPERIMENT_EXECUTION", f"target_profile:{contract.target_variable}",
                "validation_only_model_selection", "locked_test_not_used_for_selection"],
            started_at=started, completed_at=datetime.now(timezone.utc))
        self._write_artifacts(contract, result, digest, started, experiment_dir)
        failed = [name for name, record in records.items() if not record.fit_success]
        if failed and not policy.allow_partial_failure:
            raise ExperimentExecutionError("model_fit_failed:" + ",".join(failed), result)
        trace = ExperimentExecutionTrace(experiment_id=contract.experiment_id, dataset_frozen=True,
            leakage_check_passed=True, baseline_valid=True, metric_check_passed=True,
            notes=[f"dataset_sha256:{digest}", "locked_test_used_for_selection:False"])
        return result, trace

    @staticmethod
    def _regime_labels(data, *, span: int = 8, quantile: float = 0.75):
        train = np.asarray(data.y_source_train, dtype=float).reshape(-1)
        locked = np.asarray(data.y_source_locked_test, dtype=float).reshape(-1)
        if len(train) <= span or len(locked) <= span:
            raise ValueError("insufficient_samples_for_regime_assignment")
        train_slopes = (train[span:] - train[:-span]) / span
        threshold = float(np.quantile(np.abs(train_slopes), quantile))
        slopes = np.full(len(locked), np.nan)
        slopes[span:] = (locked[span:] - locked[:-span]) / span
        labels = np.full(len(locked), "unclassified", dtype=object)
        labels[span:][np.abs(slopes[span:]) <= threshold] = "steady"
        labels[span:][slopes[span:] > threshold] = "ramp_up"
        labels[span:][slopes[span:] < -threshold] = "ramp_down"
        return labels, threshold

    def _run_regime_stratified(self, contract, policy, started):
        path, digest, data = self._prepare(contract)
        experiment_dir = self.output_dir / contract.experiment_id
        models_dir = experiment_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        records = {}
        for name in tuple(contract.candidate_models):
            before = time.perf_counter()
            try:
                records[name] = self._execute_model(
                    name, contract, policy, digest, data, models_dir
                )
            except Exception as exc:
                records[name] = self._record_failure(
                    name, exc, time.perf_counter() - before, contract, digest, data
                )
        successful = [name for name, record in records.items() if record.fit_success]
        if not successful:
            raise ExperimentExecutionError("regime_all_models_failed")
        labels, threshold = self._regime_labels(data)
        regime_metrics = {}
        for name in successful:
            prediction_path = next(
                Path(item) for item in records[name].artifact_paths
                if item.endswith("_locked_test_predictions.csv")
            )
            with prediction_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            truth = np.asarray([float(row["y_true"]) for row in rows])
            prediction = np.asarray([float(row["y_pred"]) for row in rows])
            by_regime = {}
            for regime in ("ramp_up", "ramp_down", "steady"):
                mask = labels == regime
                if not np.any(mask):
                    continue
                error = prediction[mask] - truth[mask]
                by_regime[regime] = {
                    "MAE": float(np.mean(np.abs(error))),
                    "RMSE": float(np.sqrt(np.mean(error ** 2))),
                    "MBE": float(np.mean(error)),
                    "sample_count": float(np.sum(mask)),
                }
            regime_metrics[name] = by_regime
        selected = min(
            successful,
            key=lambda name: records[name].validation_metrics.get("MAE", float("inf")),
        )
        selected_regimes = regime_metrics[selected]
        if not {"ramp_up", "ramp_down"}.issubset(selected_regimes):
            raise ExperimentExecutionError("required_direction_regime_missing")
        raw_result_metrics = records[selected].locked_test_metrics
        result = ExperimentResult(
            experiment_id=contract.experiment_id,
            problem_id=contract.problem_id,
            hypothesis_id=contract.hypothesis_id,
            plan_id=contract.plan_id,
            status=ExperimentStatus.COMPLETED,
            metrics=self._canonical_metrics(raw_result_metrics),
            raw_metrics=raw_result_metrics,
            normalized_metrics=normalize_metrics(raw_result_metrics),
            baseline_metrics={"MAE": selected_regimes["ramp_down"]["MAE"]},
            candidate_locked_test_metrics={
                name: record.locked_test_metrics for name, record in records.items()
                if record.fit_success
            },
            model_records=records,
            regime_metrics=regime_metrics,
            conclusion_scope="problem_observable_premise_only",
            execution_notes=[
                "REGIME_STRATIFIED_EVALUATION",
                "train_only_slope_threshold",
                f"steady_threshold:{threshold}",
                f"validation_selected_model:{selected}",
                "mechanism_not_tested",
            ],
            started_at=started,
            completed_at=datetime.now(timezone.utc),
        )
        self._write_artifacts(contract, result, digest, started, experiment_dir)
        trace = ExperimentExecutionTrace(
            experiment_id=contract.experiment_id,
            dataset_frozen=True,
            leakage_check_passed=True,
            baseline_valid=True,
            metric_check_passed=True,
            notes=[
                f"dataset_sha256:{digest}",
                "regime_threshold_fit_on_train_only",
                "conclusion_scope:problem_observable_premise_only",
            ],
        )
        return result, trace

    @staticmethod
    def _feature_indices(features: Any) -> tuple[int, ...]:
        if not isinstance(features, list) or not features:
            raise ValueError("intervention_features_required")
        if len(features) != len(set(features)):
            raise ValueError("duplicate_intervention_features")
        indices = []
        for feature in features:
            if not isinstance(feature, str) or not feature.startswith("feature_"):
                raise ValueError(f"unknown_intervention_feature:{feature}")
            try:
                index = int(feature.split("_", 1)[1]) - 1
            except ValueError as exc:
                raise ValueError(f"unknown_intervention_feature:{feature}") from exc
            if index < 0 or index >= 30:
                raise ValueError(f"unknown_intervention_feature:{feature}")
            indices.append(index)
        return tuple(indices)

    def _validate_intervention(self, contract: ExperimentContract):
        issues = []
        control_model = contract.control.get("model") or contract.control.get("model_name")
        treatment_model = contract.treatment.get("model") or contract.treatment.get("model_name")
        if not control_model or not treatment_model:
            issues.append("intervention_group_model_required")
        elif control_model != treatment_model:
            issues.append("intervention_requires_same_model")
        if list(contract.candidate_models) != ([control_model] if control_model else []):
            issues.append("intervention_candidate_model_mismatch")
        try:
            control_features = self._feature_indices(contract.control.get("features"))
            treatment_features = self._feature_indices(contract.treatment.get("features"))
            if control_features == treatment_features:
                issues.append("control_treatment_features_must_differ")
        except ValueError as exc:
            issues.append(str(exc))
            control_features, treatment_features = (), ()
        if contract.locked_test_used_for_selection:
            issues.append("locked_test_used_for_selection")
        if issues:
            raise ValueError("invalid_feature_intervention:" + ",".join(issues))
        return control_model, control_features, treatment_features

    def _run_feature_intervention(self, contract, policy, started):
        model_name, control_features, treatment_features = self._validate_intervention(contract)
        _path, digest, control_data = self._prepare(contract, control_features)
        _path, treatment_digest, treatment_data = self._prepare(contract, treatment_features)
        if digest != treatment_digest:
            raise ValueError("intervention_dataset_identity_mismatch")
        split_names = ("train", "validation", "locked_test")
        for split in split_names:
            if not np.array_equal(control_data.source_indices[split], treatment_data.source_indices[split]):
                raise ValueError(f"intervention_split_mismatch:{split}")
            if not np.array_equal(control_data.target_indices[split], treatment_data.target_indices[split]):
                raise ValueError(f"intervention_target_alignment_mismatch:{split}")
        experiment_dir = self.output_dir / contract.experiment_id
        models_dir = experiment_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        records = {}
        for arm, data in (("control", control_data), ("treatment", treatment_data)):
            before = time.perf_counter()
            try:
                records[arm] = self._execute_model(model_name, contract, policy, digest, data, models_dir,
                    artifact_stem=f"{arm}_{model_name}")
            except Exception as exc:
                records[arm] = self._record_failure(model_name, exc, time.perf_counter() - before,
                    contract, digest, data)
        valid = all(record.fit_success for record in records.values())
        control_metrics = records["control"].locked_test_metrics
        treatment_metrics = records["treatment"].locked_test_metrics
        deltas = {
            "delta_MAE": self._metric_value(treatment_metrics, "MAE")
            - self._metric_value(control_metrics, "MAE"),
            "delta_RMSE": self._metric_value(treatment_metrics, "RMSE")
            - self._metric_value(control_metrics, "RMSE"),
            "delta_R2": self._metric_value(treatment_metrics, "R2")
            - self._metric_value(control_metrics, "R2"),
        } if valid else {}
        persistence = self.model_registry.build_adapter("persistence")
        baseline = self._canonical_metrics(
            persistence.evaluate(
                control_data.y_locked_test,
                control_data.y_source_locked_test,
            )
        )
        result = ExperimentResult(experiment_id=contract.experiment_id, problem_id=contract.problem_id,
            hypothesis_id=contract.hypothesis_id, plan_id=contract.plan_id,
            status=ExperimentStatus.COMPLETED if valid else ExperimentStatus.FAILED,
            metrics=self._canonical_metrics(treatment_metrics),
            raw_metrics=treatment_metrics,
            normalized_metrics=normalize_metrics(treatment_metrics),
            baseline_metrics=baseline,
            candidate_locked_test_metrics={"control": control_metrics, "treatment": treatment_metrics},
            model_records=records, control_metrics=control_metrics, treatment_metrics=treatment_metrics,
            metric_deltas=deltas, experiment_valid=valid,
            experiment_validity_issues=[] if valid else [f"{arm}_execution_failed" for arm, record in records.items() if not record.fit_success],
            execution_notes=["FEATURE_INTERVENTION_EXECUTION", "frozen_control_treatment_design",
                "shared_chronological_split", "locked_test_not_used_for_design_or_selection"],
            started_at=started, completed_at=datetime.now(timezone.utc))
        self._write_artifacts(contract, result, digest, started, experiment_dir)
        if not valid and not policy.allow_partial_failure:
            raise ExperimentExecutionError("feature_intervention_arm_failed", result)
        trace = ExperimentExecutionTrace(experiment_id=contract.experiment_id, dataset_frozen=True,
            leakage_check_passed=True, baseline_valid=True, metric_check_passed=valid,
            notes=[f"dataset_sha256:{digest}", "control_treatment_split_identity:True",
                "locked_test_used_for_selection:False"])
        return result, trace

    execute = run
