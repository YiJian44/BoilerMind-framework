from boilermind.core.contracts import (
    EvidenceBundle,
    HypothesisAdmissionReport,
    MechanismCritiqueDecision,
    ScientificHypothesis,
)

from boilermind.core.enums import HypothesisStatus

from .mechanism_critic import (
    evaluate_mechanism_critique,
)

from .quality_gate import (
    evaluate_hypothesis_quality,
)


def evaluate_hypothesis_admission(
    hypothesis: ScientificHypothesis,
    evidence_bundle: EvidenceBundle,
    mechanism_decision: MechanismCritiqueDecision,
) -> HypothesisAdmissionReport:
    quality_report = evaluate_hypothesis_quality(
        hypothesis,
        evidence_bundle,
    )

    mechanism_report = evaluate_mechanism_critique(
        hypothesis,
        mechanism_decision,
    )

    issues = (
        quality_report.issues
        + mechanism_report.issues
    )

    issues = list(dict.fromkeys(issues))

    passed = (
        quality_report.passed
        and mechanism_report.passed
    )

    return HypothesisAdmissionReport(
        hypothesis_id=hypothesis.hypothesis_id,
        passed=passed,
        evidence_quality_passed=(
            quality_report.passed
        ),
        mechanism_critic_passed=(
            mechanism_report.passed
        ),
        issues=issues,
    )


def admit_hypothesis(
    hypothesis: ScientificHypothesis,
    evidence_bundle: EvidenceBundle,
    mechanism_decision: MechanismCritiqueDecision,
) -> tuple[
    ScientificHypothesis,
    HypothesisAdmissionReport,
]:
    report = evaluate_hypothesis_admission(
        hypothesis,
        evidence_bundle,
        mechanism_decision,
    )

    status = (
        HypothesisStatus.QUALIFIED
        if report.passed
        else HypothesisStatus.REJECTED
    )

    updated = hypothesis.model_copy(
        update={
            "status": status,
        }
    )

    return updated, report