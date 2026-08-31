from boilermind.core.contracts import (
    EvidenceBundle,
    HypothesisQualityReport,
    ScientificHypothesis,
)

from boilermind.core.enums import (
    MechanismSupportType,
)

from .evidence_id_resolver import resolve_evidence_id


def evaluate_hypothesis_quality(
    hypothesis: ScientificHypothesis,
    evidence_bundle: EvidenceBundle,
) -> HypothesisQualityReport:
    issues: list[str] = []

    if (
        hypothesis.evidence_bundle_sha256
        != evidence_bundle.sha256
    ):
        issues.append(
            "evidence_bundle_hash_mismatch"
        )

    verified_evidence = {
        item.evidence_id: item
        for item in evidence_bundle.evidence
    }

    traceable_steps = 0

    for step in hypothesis.mechanism_steps:

        if (
            step.support_type
            == MechanismSupportType.VERIFIED_EVIDENCE
        ):
            if not step.evidence_ids:
                issues.append(
                    f"step_{step.step}_missing_evidence_id"
                )
                continue

            step_valid = True

            for evidence_id in step.evidence_ids:

                resolution = resolve_evidence_id(
                    evidence_id,
                    verified_evidence,
                )
                evidence = (
                    verified_evidence.get(resolution.resolved_id)
                    if resolution.resolved_id is not None
                    else None
                )

                if evidence is None:
                    issue = (
                        "ambiguous_evidence_id"
                        if resolution.status == "ambiguous"
                        else "unknown_evidence_id"
                    )
                    issues.append(f"{issue}:{evidence_id}")
                    step_valid = False
                    continue

                if not evidence.core_claim_eligible:
                    issues.append(
                        f"ineligible_core_evidence:{evidence_id}"
                    )
                    step_valid = False

            if step_valid:
                traceable_steps += 1

        elif (
            step.support_type
            == MechanismSupportType.DATA_OBSERVATION
        ):
            traceable_steps += 1

    total_steps = len(hypothesis.mechanism_steps)

    evidence_coverage_ratio = (
        traceable_steps / total_steps
        if total_steps
        else 0.0
    )

    if traceable_steps == 0:
        issues.append(
            "mechanism_steps_not_traceable"
        )

    passed = len(issues) == 0

    return HypothesisQualityReport(
        hypothesis_id=hypothesis.hypothesis_id,
        passed=passed,
        issues=issues,
        traceable_step_count=traceable_steps,
        total_step_count=total_steps,
        evidence_coverage_ratio=evidence_coverage_ratio,
    )


def qualify_hypotheses(
    hypotheses: list[ScientificHypothesis],
    evidence_bundle: EvidenceBundle,
) -> tuple[
    list[ScientificHypothesis],
    list[HypothesisQualityReport],
]:
    qualified: list[ScientificHypothesis] = []
    reports: list[HypothesisQualityReport] = []

    for hypothesis in hypotheses:

        report = evaluate_hypothesis_quality(
            hypothesis,
            evidence_bundle,
        )

        reports.append(report)

        if report.passed:
            qualified.append(
                hypothesis.model_copy(
                    update={
                        "status": "qualified"
                    }
                )
            )

    return qualified, reports
