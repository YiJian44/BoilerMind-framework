from __future__ import annotations

import importlib.util
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from boilermind.experiment.real_sklearn_backend import (
    RealSklearnExperimentBackend,
)

from boilermind.models.model_registry import ModelSpec


@runtime_checkable
class ModelAdapter(Protocol):
    """
    Unified model interface.

    sklearn / torch internal differences are hidden behind
    fit / predict / evaluate. Planner and Runner only see
    this interface.
    """

    def fit(
        self,
        X,
        y,
        **kwargs: Any,
    ) -> "ModelAdapter":
        ...

    def predict(self, X) -> np.ndarray:
        ...

    def evaluate(
        self,
        y_true,
        y_pred,
    ) -> dict[str, float]:
        ...


class BaseModelAdapter(ABC):
    """
    Unified adapter interface.

    fit / predict / evaluate are the ONLY entry points the
    Runner may use, regardless of sklearn / torch / legacy /
    heuristic implementation.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def fit(self, X, y, **kwargs: Any):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def evaluate(
        self,
        y_true,
        y_pred,
    ) -> dict[str, float]:
        ...


class SklearnBackendModelAdapter(BaseModelAdapter):
    """
    Adapter over the existing real_sklearn backend builders.

    fit / predict reuse the exact same sklearn estimators and
    metric computation as the production experiment backend.
    """

    __test__ = False

    def __init__(
        self,
        spec: ModelSpec,
        *,
        params: dict[str, Any] | None = None,
        random_seed: int = 42,
        backend: RealSklearnExperimentBackend | None = None,
    ):
        self.spec = spec
        self.params = params
        self.random_seed = random_seed
        self.backend = (
            backend or RealSklearnExperimentBackend()
        )
        self.estimator = None

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def fit(
        self,
        X,
        y,
        **kwargs: Any,
    ) -> "SklearnBackendModelAdapter":

        if self.spec.model_name not in (
            self.backend.SUPPORTED_MODELS
        ):
            raise ValueError(
                "model_not_supported_by_real_backend:"
                f"{self.spec.model_name}"
            )

        params = self.params

        if params is None:
            grid = self.backend._parameter_grid(
                self.spec.model_name
            )

            if not grid:
                raise RuntimeError(
                    "empty_parameter_grid:"
                    f"{self.spec.model_name}"
                )

            params = grid[0]

        self.estimator = self.backend._build_model(
            self.spec.model_name,
            dict(params),
            self.random_seed,
        )

        self.estimator.fit(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
        )

        self.params = dict(params)

        return self

    def predict(self, X) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError(
                "model_not_fitted"
            )

        predictions = np.asarray(
            self.estimator.predict(
                np.asarray(X, dtype=float)
            )
        ).reshape(-1)

        return np.maximum(0.0, predictions)

    def evaluate(
        self,
        y_true,
        y_pred,
    ) -> dict[str, float]:
        return self.backend._metrics(
            y_true,
            y_pred,
        )


class PersistenceModelAdapter(BaseModelAdapter):
    """
    Heuristic persistence baseline: last observed value
    (source mass) is the forecast.
    """

    __test__ = False

    def __init__(
        self,
        spec: ModelSpec,
        *,
        backend: RealSklearnExperimentBackend | None = None,
    ):
        self.spec = spec
        self.backend = (
            backend or RealSklearnExperimentBackend()
        )
        self._source_values: np.ndarray | None = None

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def fit(
        self,
        X,
        y,
        **kwargs: Any,
    ) -> "PersistenceModelAdapter":
        # y here is the source (forecast-origin) target value.
        self._source_values = np.asarray(
            y,
            dtype=float,
        ).reshape(-1)

        return self

    def predict(self, X) -> np.ndarray:
        if self._source_values is None:
            raise RuntimeError(
                "persistence_model_not_fitted"
            )

        return np.maximum(
            0.0,
            self._source_values,
        )

    def evaluate(
        self,
        y_true,
        y_pred,
    ) -> dict[str, float]:
        return self.backend._metrics(
            y_true,
            y_pred,
        )


class LegacyPackagedModelAdapter(BaseModelAdapter):
    """
    Inference-only adapter over the OLD project's packaged
    deep-model loader (PackagedModel), reused as-is.

    - loads config.json / feature_schema.json / scaler.joblib /
      model.pth from the old library
    - requires torch (not installed in the current venv)
    - checkpoint reuse is refused unless
      ModelSpec.checkpoint_compatibility.compatible is True
    - fit() is NOT integrated yet (old training entry exists
      in the old library; do not reimplement here)
    """

    __test__ = False

    def __init__(
        self,
        spec: ModelSpec,
        *,
        vendor_loader_path: str | Path | None = None,
    ):
        self.spec = spec

        if not spec.checkpoint_path:
            raise ValueError(
                "checkpoint_required_for_legacy_adapter:"
                f"{spec.model_name}"
            )

        if (
            spec.checkpoint_available
            and not spec.checkpoint_compatibility.get(
                "compatible",
                False,
            )
        ):
            raise ValueError(
                "checkpoint_incompatible_reuse_refused:"
                f"{spec.model_name};"
                "reuse_architecture_and_retrain_instead"
            )

        self.package_dir = Path(
            spec.checkpoint_path
        ).parent

        self.vendor_loader_path = Path(
            vendor_loader_path
            or (
                Path(__file__).resolve().parent
                / "vendor"
                / "packed_loader.py"
            )
        )

        self._packaged = None

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def _load_packaged(self):
        if self._packaged is not None:
            return self._packaged

        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "torch_required_for_legacy_deep_adapter:"
                f"{self.spec.model_name};{exc}"
            ) from exc

        if not self.vendor_loader_path.is_file():
            raise FileNotFoundError(
                self.vendor_loader_path
            )

        spec = importlib.util.spec_from_file_location(
            "boilermind_legacy_packed_loader",
            self.vendor_loader_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                "cannot_load_legacy_packed_loader"
            )

        module = importlib.util.module_from_spec(
            spec
        )

        spec.loader.exec_module(module)

        packaged = module.PackagedModel(
            self.package_dir
        )

        self._packaged = packaged

        return packaged

    def fit(
        self,
        X,
        y,
        **kwargs: Any,
    ) -> "LegacyPackagedModelAdapter":
        raise NotImplementedError(
            "training_entry_not_integrated:"
            f"{self.spec.model_name};"
            "old library train.py exists; reuse architecture "
            "and retrain through the old training entry"
        )

    def predict(self, X) -> np.ndarray:
        packaged = self._load_packaged()

        return np.asarray(
            packaged.predict(
                np.asarray(X, dtype=float)
            )
        ).reshape(-1)

    def evaluate(
        self,
        y_true,
        y_pred,
    ) -> dict[str, float]:
        backend = RealSklearnExperimentBackend()

        return backend._metrics(
            y_true,
            y_pred,
        )


# Public adapter hierarchy names (unified interface).
SklearnModelAdapter = SklearnBackendModelAdapter


class LegacyModelAdapter(LegacyPackagedModelAdapter):
    """
    Checkpoint-based legacy deep-model adapter.
    """


class TorchModelAdapter(BaseModelAdapter):
    """
    Unified source-training adapter for registered torch models.
    Torch is imported lazily, so catalog inspection works without it.
    """

    __test__ = False

    def __init__(self, spec: ModelSpec, *, config: dict[str, Any] | None = None,
                 random_seed: int = 42, device: str | None = None,
                 reuse_checkpoint: bool = False, torch_module: Any | None = None):
        self.spec = spec
        self.config = dict(config or {})
        self.random_seed = int(random_seed)
        self.requested_device = device
        self.reuse_checkpoint = bool(reuse_checkpoint)
        self._torch = torch_module
        self.model = None
        self.device = None
        self.runtime_seconds = None
        self.epochs_completed = 0
        self.best_epoch = None
        self.training_loss = None
        self.validation_loss = None
        self.warning_messages: list[str] = []
        self.target_scaling_method = "standard_train_only"
        self.target_mean_: np.ndarray | None = None
        self.target_scale_: np.ndarray | None = None

        if self.reuse_checkpoint and not (
            spec.checkpoint_available
            and spec.checkpoint_compatibility.get("compatible", False)
            and spec.can_infer_from_checkpoint
        ):
            raise ValueError(f"checkpoint_incompatible_reuse_refused:{spec.model_name}")

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    def _load_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"torch_required:torch_not_installed:{self.model_name}") from exc
        self._torch = torch
        return torch

    def _select_device(self, torch):
        cuda = bool(torch.cuda.is_available())
        if self.spec.requires_cuda and not cuda:
            raise RuntimeError(f"cuda_required_but_unavailable:{self.model_name}")
        if self.requested_device == "cuda" and not cuda:
            raise RuntimeError("cuda_requested_but_unavailable")
        if self.requested_device:
            selected = self.requested_device
        elif cuda and self.spec.gpu_supported:
            selected = "cuda"
        elif self.spec.cpu_supported:
            selected = "cpu"
        else:
            raise RuntimeError(f"no_supported_device:{self.model_name}")
        return torch.device(selected)

    def fit(self, X, y, **kwargs: Any) -> "TorchModelAdapter":
        if not self.spec.can_train_from_source:
            raise RuntimeError(f"train_from_source_not_supported:{self.model_name}")
        torch = self._load_torch()
        from boilermind.models.torch_factory import build_torch_architecture
        started = time.perf_counter()
        torch.manual_seed(self.random_seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        self.device = self._select_device(torch)
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        if X_array.ndim != 3:
            raise ValueError("torch_X_must_have_shape_samples_window_features")
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        # Fit target scaling on training targets only.  Validation/test targets
        # remain untouched and predictions are inverse-transformed in predict().
        self.target_mean_ = np.mean(y_array, axis=0, dtype=np.float64).astype(np.float32)
        self.target_scale_ = np.std(y_array, axis=0, dtype=np.float64).astype(np.float32)
        self.target_scale_ = np.where(self.target_scale_ > 1e-8, self.target_scale_, 1.0)
        y_scaled = (y_array - self.target_mean_) / self.target_scale_
        build_config = {
            "input_size": X_array.shape[-1], "window_steps": X_array.shape[1],
            "output_size": y_array.shape[-1], **self.config,
        }
        self.model = build_torch_architecture(self.model_name, torch, build_config).to(self.device)
        if self.reuse_checkpoint:
            state = torch.load(self.spec.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            scaler_path = Path(str(self.spec.checkpoint_path) + ".target_scaler.json")
            if not scaler_path.is_file():
                raise RuntimeError("checkpoint_target_scaler_missing")
            import json
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
            self.target_mean_ = np.asarray(scaler["mean"], dtype=np.float32)
            self.target_scale_ = np.asarray(scaler["scale"], dtype=np.float32)
            self.runtime_seconds = time.perf_counter() - started
            return self

        dataset = torch.utils.data.TensorDataset(torch.as_tensor(X_array), torch.as_tensor(y_scaled))
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(kwargs.get("batch_size", self.config.get("batch_size", 32))), shuffle=True, generator=generator)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(kwargs.get("learning_rate", self.config.get("learning_rate", 1e-3))))
        loss_fn = torch.nn.MSELoss()
        epochs = int(kwargs.get("epochs", self.config.get("epochs", 10)))
        patience = int(kwargs.get("patience", self.config.get("patience", epochs)))
        X_val, y_val = kwargs.get("validation_data", (None, None))
        best_loss = float("inf")
        stale = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for epoch in range(epochs):
                self.model.train()
                losses = []
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = loss_fn(self.model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))
                self.training_loss = float(np.mean(losses))
                self.epochs_completed = epoch + 1
                current = self.training_loss
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(torch.as_tensor(np.asarray(X_val, dtype=np.float32)).to(self.device))
                        raw_true = np.asarray(y_val, dtype=np.float32).reshape(pred.shape)
                        scaled_true = (raw_true - self.target_mean_) / self.target_scale_
                        true = torch.as_tensor(scaled_true).to(self.device)
                        current = float(loss_fn(pred, true).cpu().item())
                        self.validation_loss = current
                if current < best_loss:
                    best_loss, self.best_epoch, stale = current, epoch + 1, 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            self.warning_messages = [f"{type(w.message).__name__}: {w.message}" for w in caught]
        self.runtime_seconds = time.perf_counter() - started
        return self

    def predict(self, X) -> np.ndarray:
        torch = self._load_torch()
        if self.model is None:
            raise RuntimeError("model_not_fitted")
        self.model.eval()
        with torch.no_grad():
            result = self.model(torch.as_tensor(np.asarray(X, dtype=np.float32)).to(self.device))
        scaled = np.asarray(result.detach().cpu()).reshape(len(X), -1)
        if self.target_mean_ is None or self.target_scale_ is None:
            raise RuntimeError("target_scaler_not_fitted")
        restored = scaled * self.target_scale_.reshape(1, -1) + self.target_mean_.reshape(1, -1)
        return restored.squeeze(-1)

    def evaluate(self, y_true, y_pred) -> dict[str, float]:
        return RealSklearnExperimentBackend()._metrics(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1))


def build_adapter_for_spec(
    spec: ModelSpec,
    **kwargs: Any,
) -> ModelAdapter:

    if spec.model_name == "persistence":
        return PersistenceModelAdapter(
            spec,
            **kwargs,
        )

    if spec.framework == "sklearn":
        return SklearnBackendModelAdapter(
            spec,
            **kwargs,
        )

    if spec.framework == "torch":
        return TorchModelAdapter(
            spec,
            **kwargs,
        )

    raise NotImplementedError(
        "adapter_not_implemented:"
        f"{spec.model_name} "
        f"(framework={spec.framework})"
    )
