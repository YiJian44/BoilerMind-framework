from boilermind.audit.criterion_assessment import (
    CriterionAssessment,
)

from boilermind.core.contracts import (
    ExperimentAudit,
    ScientificResult,
)

from boilermind.core.enums import (
    ScientificVerdict,
)


def derive_scientific_result(
    hypothesis_id: str,
    experiment_id: str,
    audit: ExperimentAudit,
    assessment: CriterionAssessment,
) -> ScientificResult:
    if audit.experiment_id != experiment_id:
        raise ValueError(
            "Audit experiment ID mismatch."
        )

    if assessment.experiment_id != experiment_id:
        raise ValueError(
            "Criterion assessment experiment "
            "ID mismatch."
        )

    # Invalid experiment cannot support or
    # falsify a scientific hypothesis.
    if not (
        audit.execution_valid
        and audit.dataset_frozen
        and audit.leakage_check_passed
        and audit.baseline_valid
        and audit.metric_check_passed
    ):
        return ScientificResult(
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            verdict=(
                ScientificVerdict
                .INSUFFICIENT_EVIDENCE
            ),
            rationale=(
                "Experiment audit failed; "
                "no scientific conclusion allowed."
            ),
            achieved_criteria=[],
            failed_criteria=list(
                audit.issues
            ),
        )

    if (
        assessment.confirmation_met
        and not assessment.falsification_met
    ):
        verdict = ScientificVerdict.SUPPORTED

    elif (
        assessment.falsification_met
        and not assessment.confirmation_met
    ):
        verdict = ScientificVerdict.FALSIFIED

    else:
        verdict = (
            ScientificVerdict
            .INSUFFICIENT_EVIDENCE
        )

    return ScientificResult(
        hypothesis_id=hypothesis_id,
        experiment_id=experiment_id,
        verdict=verdict,
        rationale=assessment.rationale,
        achieved_criteria=list(
            assessment.achieved_criteria
        ),
        failed_criteria=list(
            assessment.failed_criteria
        ),
    )