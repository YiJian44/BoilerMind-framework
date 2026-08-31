from __future__ import annotations

from typing import Any

from boilermind.experiment.real_sklearn_backend import (
    RealSklearnExperimentBackend,
)


class ExperimentRunner:
    """
    BoilerMind real experiment runner.

    Current supported backend:
    - real_sklearn

    No mock metrics.
    No random input.
    """

    def __init__(self):
        self.real_sklearn = RealSklearnExperimentBackend()

    def run(
        self,
        contract: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(contract, dict):
            raise TypeError(
                "experiment_contract_must_be_dict"
            )

        backend = contract.get(
            "execution_backend",
            "real_sklearn",
        )

        if backend != "real_sklearn":
            raise ValueError(
                f"unsupported_execution_backend:{backend}"
            )

        if not contract.get("dataset_path"):
            raise ValueError(
                "dataset_path_required"
            )

        if not contract.get("model_candidates"):
            raise ValueError(
                "model_candidates_required"
            )

        return self.real_sklearn.run(
            contract
        )
