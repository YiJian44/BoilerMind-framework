from __future__ import annotations

from dataclasses import dataclass

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
    ScientificResult,
)
from boilermind.core.enums import ExperimentStatus, ScientificVerdict

from .schemas import PerformanceAnalysis


_ERROR_METRICS = ("MAE", "RMSE")


@dataclass(frozen=True)
class ExperimentReflection:
    """Read-only deterministic interpretation of already-produced metrics."""

    performance_analysis: PerformanceAnalysis
    should_adjust: bool


def _candidate_metric(
    contract: ExperimentContract,
    result: ExperimentResult,
    metric: str,
) -> float | None:
    canonical = result.normalized_metrics.get(metric)
    if isinstance(canonical, (int, float)):
        return float(canonical)
    direct = result.metrics.get(metric)
    if direct is not None:
        return float(direct)
    if len(contract.candidate_models) == 1:
        value = result.candidate_locked_test_metrics.get(
            contract.candidate_models[0], {}
        ).get(metric)
        if value is not None:
            return float(value)
    return None


def reflect_experiment(
    contract: ExperimentContract,
    result: ExperimentResult,
    scientific_result: ScientificResult,
) -> ExperimentReflection:
    """Compare real candidate error with a real baseline; never invent a target."""
    if result.experiment_id != contract.experiment_id:
        raise ValueError("experiment_id_mismatch")
    if scientific_result.experiment_id != result.experiment_id:
        raise ValueError("scientific_result_experiment_id_mismatch")
    if result.hypothesis_id != contract.hypothesis_id:
        raise ValueError("hypothesis_id_mismatch")
    if scientific_result.hypothesis_id != result.hypothesis_id:
        raise ValueError("scientific_result_hypothesis_id_mismatch")

    if result.status != ExperimentStatus.COMPLETED:
        analysis = PerformanceAnalysis(
            status="execution_failed",
            evidence=[f"experiment_status:{result.status.value}"],
        )
        return ExperimentReflection(analysis, False)

    for metric in _ERROR_METRICS:
        observed = _candidate_metric(contract, result, metric)
        baseline = result.baseline_metrics.get(metric)
        if observed is None or baseline is None:
            continue
        observed = float(observed)
        baseline = float(baseline)
        relative = None if baseline == 0 else (observed - baseline) / abs(baseline)
        high_error = observed >= baseline
        analysis = PerformanceAnalysis(
            status=("high_error_relative_to_baseline" if high_error else "stable"),
            metric=metric,
            observed_value=observed,
            baseline_value=baseline,
            relative_change=relative,
            evidence=[
                f"candidate_{metric}:{observed}",
                f"baseline_{metric}:{baseline}",
                f"scientific_verdict:{scientific_result.verdict.value}",
            ],
        )
        return ExperimentReflection(analysis, high_error)

    stable_verdict = scientific_result.verdict in {
        ScientificVerdict.SUPPORTED,
        ScientificVerdict.PARTIALLY_SUPPORTED,
    }
    analysis = PerformanceAnalysis(
        status="stable" if stable_verdict else "not_comparable",
        evidence=[
            "no_candidate_baseline_error_pair",
            f"scientific_verdict:{scientific_result.verdict.value}",
        ],
    )
    return ExperimentReflection(analysis, False)
