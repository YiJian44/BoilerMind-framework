from dataclasses import dataclass
from datetime import datetime, timezone

from boilermind.audit.execution_trace import (
    ExperimentExecutionTrace,
)

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
)

from boilermind.core.enums import ExperimentStatus
from boilermind.experiment.metric_normalizer import normalize_metrics


@dataclass(frozen=True)
class TestOnlyOutcome:
    __test__ = False

    hypothesis_id: str

    metrics: dict[str, float]

    baseline_metrics: dict[str, float]

    dataset_frozen: bool = True
    leakage_check_passed: bool = True
    baseline_valid: bool = True
    metric_check_passed: bool = True


class TestOnlyExperimentRunner:
    __test__ = False

    """
    TEST-ONLY runner.

    This class must never be used as formal scientific
    evidence or written into production research history.
    """

    is_test_only = True

    def __init__(
        self,
        outcomes: dict[str, TestOnlyOutcome],
    ):
        self._outcomes = outcomes

    def run(
        self,
        contract: ExperimentContract,
    ) -> tuple[
        ExperimentResult,
        ExperimentExecutionTrace,
    ]:
        outcome = self._outcomes.get(
            contract.hypothesis_id
        )

        if outcome is None:
            raise ValueError(
                f"No TEST-ONLY outcome configured for "
                f"{contract.hypothesis_id}"
            )

        if (
            outcome.hypothesis_id
            != contract.hypothesis_id
        ):
            raise ValueError(
                "TEST-ONLY outcome hypothesis mismatch."
            )

        started_at = datetime.now(
            timezone.utc
        )

        result = ExperimentResult(
            experiment_id=contract.experiment_id,
            hypothesis_id=contract.hypothesis_id,
            status=ExperimentStatus.COMPLETED,
            metrics=dict(outcome.metrics),
            raw_metrics=dict(outcome.metrics),
            normalized_metrics=normalize_metrics(outcome.metrics),
            baseline_metrics=dict(
                outcome.baseline_metrics
            ),
            artifacts=[],
            execution_notes=[
                "TEST_ONLY_EXECUTION",
                "Not valid as formal scientific evidence.",
            ],
            started_at=started_at,
            completed_at=datetime.now(
                timezone.utc
            ),
        )

        trace = ExperimentExecutionTrace(
            experiment_id=contract.experiment_id,
            dataset_frozen=(
                outcome.dataset_frozen
            ),
            leakage_check_passed=(
                outcome.leakage_check_passed
            ),
            baseline_valid=(
                outcome.baseline_valid
            ),
            metric_check_passed=(
                outcome.metric_check_passed
            ),
            notes=[
                "TEST_ONLY_EXECUTION_TRACE"
            ],
        )

        return result, trace
