from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from boilermind.experiment.time_series_data import DatasetBuilder, TimeSeriesDataContract
from boilermind.models.catalog import build_default_registry


class TorchExperimentBackend:
    """Contract-driven backend; model-specific choices stay in the registry/factory."""

    def __init__(self, *, model_registry=None, dataset_builder=None):
        self.model_registry = model_registry or build_default_registry()
        self.dataset_builder = dataset_builder or DatasetBuilder()

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def run(self, contract: dict[str, Any]) -> dict[str, Any]:
        path = Path(contract["dataset_path"])
        data = self.dataset_builder.build_from_csv(path, TimeSeriesDataContract(
            feature_columns=tuple(range(30)), target_columns=(30,),
            sampling_interval_seconds=int(contract["sampling_interval_seconds"]),
            window_steps=int(contract["window_steps"]), prediction_horizon=int(contract["prediction_horizon_steps"]),
            train_ratio=float(contract["train_ratio"]), validation_ratio=float(contract["validation_ratio"]),
        ))
        digest = self._hash(path)
        records, scores = {}, {}
        for name in contract["model_candidates"]:
            adapter = self.model_registry.build_adapter(name, random_seed=int(contract["random_seed"]))
            adapter.fit(data.X_train, data.y_train, validation_data=(data.X_validation, data.y_validation))
            validation = adapter.evaluate(data.y_validation, adapter.predict(data.X_validation))
            locked = adapter.evaluate(data.y_locked_test, adapter.predict(data.X_locked_test))
            scores[name] = validation["mae_t_h"]
            records[name] = {
                "fit_success": True, "fit_converged": True, "warnings": list(adapter.warning_messages),
                "failure_reason": None, "elapsed_seconds": adapter.runtime_seconds, "model_config": dict(adapter.config),
                "validation_metrics": validation, "locked_test_metrics": locked,
                "train_samples": len(data.X_train), "validation_samples": len(data.X_validation),
                "test_samples": len(data.X_locked_test), "random_seed": int(contract["random_seed"]),
                "dataset_sha256": digest, "artifact_paths": [], "device": str(adapter.device),
                "epochs_completed": adapter.epochs_completed, "best_epoch": adapter.best_epoch,
                "training_loss": adapter.training_loss, "validation_loss": adapter.validation_loss,
            }
        selected = min(scores, key=scores.get)
        metric_adapter = self.model_registry.build_adapter(contract["model_candidates"][0])
        persistence = np.repeat(data.y_validation[-1:], len(data.y_locked_test), axis=0)
        return {
            "experiment_id": contract["experiment_id"], "status": "completed", "dataset": {"sha256": digest},
            "models": records, "selected_model_by_validation": selected,
            "reference_model": {"locked_test_metrics": metric_adapter.evaluate(data.y_locked_test, persistence)},
            "split": {"locked_test_used_for_selection": False},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
