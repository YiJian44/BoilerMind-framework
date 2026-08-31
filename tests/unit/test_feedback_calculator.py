from datetime import datetime, timezone

from boilermind.core.contracts import (
    ExperimentAudit,
    ExperimentResult,
    ScientificResult,
)

from boilermind.core.enums import (
    ExperimentStatus,
    ScientificVerdict,
)

from boilermind.ranking.feedback_calculator import (
    calculate_experiment_feedback,
    calculate_metric_effect,
)


def make_result(
    *,
    candidate_mae=0.020,
    baseline_mae=0.030,
    test_only=False,
):
    notes = []

    if test_only:
        notes.append(
            "TEST_ONLY_EXECUTION"
        )

    return ExperimentResult(
        experiment_id="EXP-001",
        hypothesis_id="H001",
        status=ExperimentStatus.COMPLETED,
        metrics={
            "MAE": candidate_mae,
        },
        baseline_metrics={
            "MAE": baseline_mae,
        },
        artifacts=[],
        execution_notes=notes,
        started_at=datetime.now(
            timezone.utc
        ),
        completed_at=datetime.now(
            timezone.utc
        ),
    )


def make_valid_audit():
    return ExperimentAudit(
        experiment_id="EXP-001",
        execution_valid=True,
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
        issues=[],
    )


def make_scientific_result(
    verdict,
):
    return ScientificResult(
        hypothesis_id="H001",
        experiment_id="EXP-001",
        verdict=verdict,
        rationale="Test verdict.",
        achieved_criteria=[],
        failed_criteria=[],
    )


def test_metric_effect_positive_when_mae_improves():
    effect = calculate_metric_effect(
        make_result(
            candidate_mae=0.020,
            baseline_mae=0.030,
        )
    )

    assert effect > 0


def test_metric_effect_negative_when_mae_worsens():
    effect = calculate_metric_effect(
        make_result(
            candidate_mae=0.040,
            baseline_mae=0.030,
        )
    )

    assert effect < 0


def test_supported_result_produces_positive_feedback():
    feedback = (
        calculate_experiment_feedback(
            make_result(),
            make_valid_audit(),
            make_scientific_result(
                ScientificVerdict.SUPPORTED
            ),
        )
    )

    assert feedback.priority_delta > 0


def test_falsified_result_produces_negative_feedback():
    feedback = (
        calculate_experiment_feedback(
            make_result(
                candidate_mae=0.040,
                baseline_mae=0.030,
            ),
            make_valid_audit(),
            make_scientific_result(
                ScientificVerdict.FALSIFIED
            ),
        )
    )

    assert feedback.priority_delta < 0


def test_test_only_feedback_has_reduced_strength():
    feedback = (
        calculate_experiment_feedback(
            make_result(
                test_only=True
            ),
            make_valid_audit(),
            make_scientific_result(
                ScientificVerdict.FALSIFIED
            ),
        )
    )

    assert feedback.evidence_strength == 0.0


def test_invalid_experiment_does_not_change_ranking():
    audit = make_valid_audit().model_copy(
        update={
            "execution_valid": False,
            "leakage_check_passed": False,
        }
    )

    feedback = (
        calculate_experiment_feedback(
            make_result(),
            audit,
            make_scientific_result(
                ScientificVerdict
                .INSUFFICIENT_EVIDENCE
            ),
        )
    )

    assert feedback.priority_delta == 0.0
    assert feedback.evidence_strength == 0.0
