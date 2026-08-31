from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_31v_library import protocol_dict, protocol_sha256  # noqa: E402
from v31_common import build_sklearn_estimator  # noqa: E402


def test_rf_receives_seed_and_cpu_limit() -> None:
    estimator = build_sklearn_estimator(
        "rf", {"n_estimators": 5, "max_depth": 3}, seed=19, n_jobs=2
    )

    assert estimator.random_state == 19
    assert estimator.n_jobs == 2


def test_rf_same_seed_repeats_and_different_seed_changes_predictions() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(160, 8))
    y = X[:, 0] * 0.4 - X[:, 1] * 0.2 + rng.normal(size=160)

    def fit(seed: int):
        return build_sklearn_estimator(
            "rf", {"n_estimators": 20, "max_depth": 5}, seed=seed, n_jobs=2
        ).fit(X, y)

    prediction_a = fit(7).predict(X)
    prediction_b = fit(7).predict(X)
    prediction_c = fit(19).predict(X)

    assert np.allclose(prediction_a, prediction_b, rtol=0.0, atol=1e-14)
    assert not np.allclose(prediction_a, prediction_c, rtol=0.0, atol=1e-14)


def test_seed_and_parallel_controls_are_part_of_protocol_hash() -> None:
    base = {
        "horizon": 40,
        "max_epochs": 100,
        "patience": 15,
        "device": "cuda",
        "sklearn_n_jobs": 2,
        "parallel_execution": True,
    }
    protocol_7 = protocol_dict(seed=7, **base)
    protocol_19 = protocol_dict(seed=19, **base)

    assert protocol_7["sklearn_n_jobs"] == 2
    assert protocol_7["parallel_execution"] is True
    assert protocol_sha256(protocol_7) != protocol_sha256(protocol_19)
