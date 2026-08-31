from boilermind.core.contracts import (
    ExperimentAudit,
    ExperimentResult,
    ScientificResult,
)

from boilermind.core.enums import (
    ScientificVerdict,
)

from boilermind.ranking.dynamic_ranker import (
    ExperimentFeedback,
)


LOWER_IS_BETTER = {
    "MAE",
    "RMSE",
    "MSE",
    "MAPE",
}

HIGHER_IS_BETTER = {
    "R2",
    "R²",
}


def _clamp(
    value: float,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    return max(
        lower,
        min(upper, value),
    )


def _metric_improvement(
    metric: str,
    candidate_value: float,
    baseline_value: float,
) -> float | None:
    """
    Return normalized improvement relative to baseline.

    Positive:
        candidate is better than baseline.

    Negative:
        candidate is worse than baseline.
    """

    denominator = max(
        abs(baseline_value),
        1e-12,
    )

    if metric in LOWER_IS_BETTER:
        improvement = (
            baseline_value
            - candidate_value
        ) / denominator

    elif metric in HIGHER_IS_BETTER:
        improvement = (
            candidate_value
            - baseline_value
        ) / denominator

    else:
        return None

    return _clamp(improvement)


def calculate_metric_effect(
    result: ExperimentResult,
    primary_metric: str | None = None,
) -> float:
    """
    Aggregate candidate-vs-baseline metric improvement.

    This is NOT a scientific verdict.
    It only measures experimental performance direction.
    """

    effects: dict[str, float] = {}

    for metric, candidate_value in (
        result.metrics.items()
    ):
        if metric not in result.baseline_metrics:
            continue

        effect = _metric_improvement(
            metric=metric,
            candidate_value=candidate_value,
            baseline_value=(
                result.baseline_metrics[metric]
            ),
        )

        if effect is not None:
            effects[metric] = effect

    if not effects:
        return 0.0

    if primary_metric and primary_metric in effects:
        secondary = [value for name, value in effects.items() if name != primary_metric]
        secondary_mean = sum(secondary) / len(secondary) if secondary else effects[primary_metric]
        return round(0.70 * effects[primary_metric] + 0.30 * secondary_mean, 6)
    return round(sum(effects.values()) / len(effects), 6)


def calculate_experiment_feedback(
    result: ExperimentResult,
    audit: ExperimentAudit,
    scientific_result: ScientificResult,
    *,
    primary_metric: str | None = None,
) -> ExperimentFeedback:
    """
    Convert audited experimental evidence into ranking feedback.

    The ranker does not decide scientific truth.
    It only receives a signed experimental feedback signal.
    """

    if (
        result.experiment_id
        != audit.experiment_id
    ):
        raise ValueError(
            "Result/Audit experiment ID mismatch."
        )

    if (
        result.experiment_id
        != scientific_result.experiment_id
    ):
        raise ValueError(
            "Result/ScientificResult experiment "
            "ID mismatch."
        )

    if (
        result.hypothesis_id
        != scientific_result.hypothesis_id
    ):
        raise ValueError(
            "Hypothesis ID mismatch."
        )

    verdict = scientific_result.verdict

    # Invalid or insufficient experiments must not
    # change scientific validation priority.
    if (
        not audit.execution_valid
        or verdict
        == ScientificVerdict.INSUFFICIENT_EVIDENCE
    ):
        return ExperimentFeedback(
            hypothesis_id=result.hypothesis_id,
            experiment_id=result.experiment_id,
            verdict=(
                ScientificVerdict
                .INSUFFICIENT_EVIDENCE
            ),
            priority_delta=0.0,
            evidence_strength=0.0,
            rationale=(
                "Experiment cannot provide valid "
                "ranking feedback because evidence "
                "is insufficient or audit failed."
            ),
        )

    metric_effect = calculate_metric_effect(result, primary_metric)

    # TEST-ONLY runs are deliberately weakened.
    is_test_only = (
        "TEST_ONLY_EXECUTION"
        in result.execution_notes
    )

    is_mock_or_replay = any(
        marker in result.execution_notes
        for marker in ("MOCK_EXECUTION", "ARTIFACT_REPLAY", "REPLAY_EXECUTION")
    )
    evidence_strength = 0.0 if is_test_only or is_mock_or_replay else 1.0

    if verdict == ScientificVerdict.SUPPORTED:
        # Positive experimental feedback.
        # Better performance produces stronger support.
        delta = (
            0.10
            + 0.15
            * max(0.0, metric_effect)
        )

        delta = min(
            delta,
            0.25,
        )

    elif verdict == ScientificVerdict.FALSIFIED:
        # Negative feedback.
        # Strongly worse experimental performance
        # produces a larger downward adjustment.
        delta = -(
            0.10
            + 0.20
            * max(0.0, -metric_effect)
        )

        delta = max(
            delta,
            -0.30,
        )

    elif verdict == ScientificVerdict.PARTIALLY_SUPPORTED:
        delta = (
            0.05
            * max(
                0.0,
                metric_effect,
            )
        )

    else:
        delta = 0.0

    return ExperimentFeedback(
        hypothesis_id=result.hypothesis_id,
        experiment_id=result.experiment_id,
        verdict=verdict,
        priority_delta=round(
            delta,
            6,
        ),
        evidence_strength=evidence_strength,
        rationale=(
            "Ranking feedback calculated from "
            f"scientific verdict={verdict.value}, "
            f"metric_effect={metric_effect:.6f}, "
            f"test_only={is_test_only}."
        ),
    )
