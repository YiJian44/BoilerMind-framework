from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeSeriesDataContract:
    feature_columns: tuple[int | str, ...]
    target_columns: tuple[int | str, ...]
    sampling_interval_seconds: int
    window_steps: int
    prediction_horizon: int
    train_ratio: float = 0.70
    validation_ratio: float = 0.15


@dataclass(frozen=True)
class TimeSeriesDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_locked_test: np.ndarray
    y_locked_test: np.ndarray
    y_source_train: np.ndarray
    y_source_validation: np.ndarray
    y_source_locked_test: np.ndarray
    scaler: Any
    source_indices: dict[str, np.ndarray]
    target_indices: dict[str, np.ndarray]


class DatasetBuilder:
    def build_from_csv(self, path: str | Path, contract: TimeSeriesDataContract) -> TimeSeriesDataset:
        import pandas as pd
        frame = pd.read_csv(path, header=None)
        return self.build(frame, contract)

    def build(self, data: Any, contract: TimeSeriesDataContract) -> TimeSeriesDataset:
        from sklearn.preprocessing import MinMaxScaler

        if hasattr(data, "loc"):
            features = data.loc[:, list(contract.feature_columns)].to_numpy(dtype=float)
            targets = data.loc[:, list(contract.target_columns)].to_numpy(dtype=float)
        else:
            values = np.asarray(data, dtype=float)
            features = values[:, [int(c) for c in contract.feature_columns]]
            targets = values[:, [int(c) for c in contract.target_columns]]

        source = np.arange(contract.window_steps - 1, len(features) - contract.prediction_horizon)
        target = source + contract.prediction_horizon
        count = len(source)
        train_end = int(count * contract.train_ratio)
        validation_end = train_end + int(count * contract.validation_ratio)
        if train_end < 1 or validation_end <= train_end or validation_end >= count:
            raise ValueError("invalid_chronological_split")

        split = {
            "train": np.arange(0, train_end),
            "validation": np.arange(train_end, validation_end),
            "locked_test": np.arange(validation_end, count),
        }
        last_train_origin = int(source[split["train"][-1]])
        scaler = MinMaxScaler().fit(features[: last_train_origin + 1])
        scaled = scaler.transform(features)
        X = np.stack([scaled[i - contract.window_steps + 1:i + 1] for i in source])
        y = targets[target]

        return TimeSeriesDataset(
            X_train=X[split["train"]], y_train=y[split["train"]],
            X_validation=X[split["validation"]], y_validation=y[split["validation"]],
            X_locked_test=X[split["locked_test"]], y_locked_test=y[split["locked_test"]],
            y_source_train=targets[source][split["train"]],
            y_source_validation=targets[source][split["validation"]],
            y_source_locked_test=targets[source][split["locked_test"]],
            scaler=scaler,
            source_indices={name: source[idx] for name, idx in split.items()},
            target_indices={name: target[idx] for name, idx in split.items()},
        )
