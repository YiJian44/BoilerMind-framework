"""Shared helpers for 31-feature -> V(m^3/s) direct soft-sensing training.

Self-contained and additive: does not touch the existing 30/M@40 framework
code under src/boilermind. Designed to run identically on a local CPU box and
on the Aliyun T4 server (numpy/pandas/scikit-learn/torch only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

R = 0.461526  # ideal gas constant, kJ/(kg.K)

# 31 authoritative features, 1-based boiler_181var columns (configs/real_prediction_dev.json)
SOFT_SENSOR_FEATURES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 15, 18, 26, 30, 31, 33, 34,
    100, 101, 102, 103, 108, 109, 111, 112, 117, 118, 125, 126, 168, 175,
]
MASS_COL = 16
PRESSURE_COL = 1
TEMPERATURE_COL = 9

WINDOW = 20
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10

REPO_ROOT = Path(__file__).resolve().parent.parent


def specific_volume(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """m^3/kg, ideal gas, absolute pressure MPa."""
    return R * (T + 273.15) / (P * 1000.0)


def volume_flow(M: np.ndarray, P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Steam volumetric flow m^3/s from mass flow t/h, absolute P, T."""
    return M * (1000.0 / 3600.0) * specific_volume(P, T)


def load_181_frame(path: str | Path):
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, header=0)
    return pd.read_csv(path, header=0)


def build_dataset(
    path: str | Path,
    *,
    horizon: int,
    window: int = WINDOW,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
):
    """Build windowed 31-feature / V-target tensors with a chronological split.

    Returns a dict with X/y per split (X shaped (N, window, 31)), the feature
    scaler, source/target indices, and y_source (for the persistence baseline).
    Feature scaling is fit on train-origin rows only (leakage-safe).
    """
    from sklearn.preprocessing import MinMaxScaler

    df = load_181_frame(path)
    feat_cols = [str(c) for c in SOFT_SENSOR_FEATURES]
    X_raw = df.loc[:, feat_cols].to_numpy(dtype=float)
    M = df[str(MASS_COL)].to_numpy(dtype=float)
    P = df[str(PRESSURE_COL)].to_numpy(dtype=float)
    T = df[str(TEMPERATURE_COL)].to_numpy(dtype=float)
    yV = volume_flow(M, P, T)

    n = len(df)
    source = np.arange(window - 1, n - horizon)
    target = source + horizon
    count = len(source)
    train_end = int(count * train_ratio)
    validation_end = train_end + int(count * validation_ratio)
    if train_end < window or validation_end <= train_end or validation_end >= count:
        raise ValueError("invalid_chronological_split")

    last_train_origin = int(source[train_end - 1])
    scaler = MinMaxScaler().fit(X_raw[: last_train_origin + 1])
    Xs = scaler.transform(X_raw)
    X = np.stack([Xs[i - window + 1 : i + 1] for i in source])  # (N, window, 31)
    y = yV[target]
    y_source = yV[source]

    split = {
        "train": np.arange(0, train_end),
        "validation": np.arange(train_end, validation_end),
        "locked_test": np.arange(validation_end, count),
    }
    return {
        "X": X,
        "y": y,
        "y_source": y_source,
        "source_indices": source,
        "target_indices": target,
        "split": split,
        "scaler": scaler,
        "n_total": count,
    }


def split_arrays(data: dict[str, np.ndarray | dict], split: dict[str, np.ndarray], key: str):
    out = {}
    for name, idx in split.items():
        out[name] = data[key][idx]
    return out


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    return {
        "mae_m3_s": float(mean_absolute_error(actual, predicted)),
        "rmse_m3_s": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mbe_m3_s": float(np.mean(predicted - actual)),
    }


# ---------------------------------------------------------------------------
# Deep model implementations (self-contained)
# ---------------------------------------------------------------------------

def build_deep_module(name: str, n_features: int, window: int, device: str):
    """Return an nn.Module with a single ``(N, window, features) -> (N, 1)`` forward."""
    if name == "lstm":
        return _RNNSensor(nn.LSTM, n_features, 64, 2, 0.1).to(device)
    if name == "gru":
        return _RNNSensor(nn.GRU, n_features, 64, 2, 0.1).to(device)
    if name == "transformer":
        return _TransformerSensor(n_features, window, 64, 4, 2, 128, 0.1).to(device)
    if name == "dlinear":
        return _DLinearSensor(n_features, window, 5).to(device)
    raise ValueError(f"unknown_deep_model:{name}")


class _RNNSensor(torch.nn.Module):
    def __init__(self, cell, n_features, hidden, n_layers, dropout):
        super().__init__()
        self.rnn = cell(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class _TransformerSensor(torch.nn.Module):
    def __init__(self, n_features, window, d_model, nhead, n_layers, dim_ff, dropout):
        super().__init__()
        self.embed = torch.nn.Linear(n_features, d_model)
        self.pos = torch.nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(layer, n_layers)
        self.head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        e = self.embed(x) + self.pos
        return self.head(self.encoder(e)[:, -1, :])


class _DLinearSensor(torch.nn.Module):
    def __init__(self, n_features, window, kernel=5):
        super().__init__()
        self.trend = torch.nn.Linear(window * n_features, 1)
        self.residual = torch.nn.Linear(window * n_features, 1)
        self.kernel = kernel

    def forward(self, x):
        B, W, F = x.shape
        pad = self.kernel // 2
        padded = torch.cat(
            [x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, pad, -1)], dim=1
        )
        trend = padded.unfold(1, self.kernel, 1).mean(-1)
        residual = x - trend
        return self.trend(trend.reshape(B, -1)) + self.residual(residual.reshape(B, -1))


class TorchSensor:
    """Sklearn-style trainable wrapper for the deep modules (target z-score
    train-only + inverse transform + early stopping)."""

    def __init__(
        self,
        name: str,
        *,
        max_epochs: int = 100,
        patience: int = 15,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.name = name
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed)
        self.model = None
        self.config = {
            "architecture": name,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
        }

    def fit(self, X, y, X_val, y_val) -> "TorchSensor":
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        n_features = X.shape[-1]
        window = X.shape[1]
        self.model = build_deep_module(self.name, n_features, window, self.device)

        y_mean = float(np.mean(y))
        y_std = float(max(np.std(y), 1e-6))
        self.y_mean_ = y_mean
        self.y_std_ = y_std
        ys = (y - y_mean) / y_std
        y_val_s = (y_val - y_mean) / y_std

        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        yt = torch.from_numpy(ys.astype(np.float32)).reshape(-1, 1).to(self.device)
        Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(self.device)
        yv = torch.from_numpy(y_val_s.astype(np.float32)).reshape(-1, 1).to(self.device)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(self.batch_size, len(Xt)),
            shuffle=False,
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss()

        best = float("inf")
        best_state = None
        best_epoch = 0
        stale = 0
        self.epochs_completed = 0
        self.training_history_ = []
        for epoch in range(self.max_epochs):
            self.model.train()
            batch_losses = []
            for xb, yb in loader:
                opt.zero_grad()
                loss = lossf(self.model(xb), yb)
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.item()))
            train_loss = float(np.mean(batch_losses))
            self.model.eval()
            with torch.no_grad():
                vloss = float(lossf(self.model(Xv), yv).item())
            self.epochs_completed = epoch + 1
            self.training_history_.append(
                {"epoch": epoch, "train_loss": train_loss, "validation_loss": vloss}
            )
            if vloss < best - 1e-8:
                best = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_validation_loss_ = best
        self.best_epoch_ = best_epoch
        self.model.eval()
        self.runtime_seconds = getattr(self, "runtime_seconds", None)
        return self

    def predict(self, X) -> np.ndarray:
        import torch

        self.model.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(Xt).detach().cpu().numpy().reshape(-1)
        return out * self.y_std_ + self.y_mean_


# ---------------------------------------------------------------------------
# sklearn estimator builders (grids follow the framework real_sklearn_backend)
# ---------------------------------------------------------------------------

SKLEARN_MODELS = [
    "ridge",
    "bayesianridge",
    "elasticnet",
    "pls",
    "svr",
    "rf",
    "mlp",
    "knn",
    "hgb",
]
DEEP_MODELS = [
    "transformer",
    "lstm",
    "dlinear",
    "gru",
]
ALL_MODELS = ["persistence"] + SKLEARN_MODELS + DEEP_MODELS


def sklearn_grid(model_id: str) -> list[dict[str, Any]]:
    # 各 sklearn 模型统一 4 个候选，覆盖超参谷底（MAE 先降后升），保证调参轮次公平
    grids = {
        "ridge": [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
        "bayesianridge": [
            {"alpha_1": 1e-6}, {"alpha_1": 1e-4}, {"alpha_1": 1e-2}, {"alpha_1": 1.0},
        ],
        "elasticnet": [
            {"alpha": 0.001, "l1_ratio": 0.3},
            {"alpha": 0.01, "l1_ratio": 0.5},
            {"alpha": 0.1, "l1_ratio": 0.5},
            {"alpha": 1.0, "l1_ratio": 0.7},
        ],
        "pls": [
            {"n_components": 2}, {"n_components": 4}, {"n_components": 6}, {"n_components": 8},
        ],
        "svr": [
            {"C": 1.0, "epsilon": 0.05, "kernel": "rbf"},
            {"C": 10.0, "epsilon": 0.05, "kernel": "rbf"},
            {"C": 100.0, "epsilon": 0.05, "kernel": "rbf"},
            {"C": 10.0, "epsilon": 0.1, "kernel": "rbf"},
        ],
        "rf": [
            {"n_estimators": 50, "max_depth": 10},
            {"n_estimators": 100, "max_depth": 15},
            {"n_estimators": 150, "max_depth": 20},
            {"n_estimators": 200, "max_depth": None},
        ],
        "mlp": [
            {"hidden_layer_sizes": (32,), "max_iter": 300},
            {"hidden_layer_sizes": (64,), "max_iter": 300},
            {"hidden_layer_sizes": (128,), "max_iter": 300},
            {"hidden_layer_sizes": (64, 32), "max_iter": 500},
        ],
        "knn": [
            {"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 9}, {"n_neighbors": 15},
        ],
        "hgb": [
            {"learning_rate": 0.01, "max_iter": 200, "max_leaf_nodes": 31},
            {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 31},
            {"learning_rate": 0.10, "max_iter": 150, "max_leaf_nodes": 31},
            {"learning_rate": 0.20, "max_iter": 100, "max_leaf_nodes": 31},
        ],
        "gpr": [{}],
    }
    if model_id not in grids:
        raise ValueError(f"unsupported_model:{model_id}")
    return grids[model_id]


def torch_grid(model_id: str) -> list[dict[str, Any]]:
    """torch 模型超参网格（lr 4 档，与 sklearn 4 候选对齐，保证调参轮次公平）。"""
    grids = {
        "lstm": [{"lr": 5e-4}, {"lr": 1e-3}, {"lr": 2e-3}, {"lr": 4e-3}],
        "gru": [{"lr": 5e-4}, {"lr": 1e-3}, {"lr": 2e-3}, {"lr": 4e-3}],
        "transformer": [{"lr": 5e-4}, {"lr": 1e-3}, {"lr": 2e-3}],
        "dlinear": [{"lr": 5e-4}, {"lr": 1e-3}, {"lr": 2e-3}],
    }
    if model_id not in grids:
        raise ValueError(f"unsupported_torch_model:{model_id}")
    return grids[model_id]


def build_sklearn_estimator(
    model_id: str,
    params: dict[str, Any],
    *,
    seed: int = 42,
    n_jobs: int = -1,
):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    if model_id == "ridge":
        return Ridge(**params)
    if model_id == "bayesianridge":
        return BayesianRidge(**params)
    if model_id == "elasticnet":
        return ElasticNet(**params, random_state=seed, max_iter=5000)
    if model_id == "pls":
        return PLSRegression(**params)
    if model_id == "svr":
        return SVR(**params)
    if model_id == "rf":
        return RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
    if model_id == "mlp":
        return MLPRegressor(**params, random_state=seed, early_stopping=True)
    if model_id == "knn":
        return KNeighborsRegressor(**params, n_jobs=n_jobs)
    if model_id == "hgb":
        return HistGradientBoostingRegressor(**params, random_state=seed)
    if model_id == "gpr":
        return GaussianProcessRegressor(**params, random_state=seed)
    raise ValueError(f"unsupported_model:{model_id}")
