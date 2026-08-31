from boilermind.core.contracts import (
    MechanismCritiqueDecision,
    MechanismCritiqueReport,
    ScientificHypothesis,
)


def evaluate_mechanism_critique(
    hypothesis: ScientificHypothesis,
    decision: MechanismCritiqueDecision,
) -> MechanismCritiqueReport:
    if (
        hypothesis.hypothesis_id
        != decision.hypothesis_id
    ):
        raise ValueError(
            "Mechanism critique hypothesis ID mismatch."
        )

    issues = list(decision.issues)

    if not decision.causal_chain_complete:
        issues.append(
            "causal_chain_incomplete"
        )

    if not decision.physical_consistency:
        issues.append(
            "physical_inconsistency"
        )

    if not decision.temporal_consistency:
        issues.append(
            "temporal_inconsistency"
        )

    if not decision.scope_consistency:
        issues.append(
            "scope_inconsistency"
        )

    if not decision.single_testable_claim:
        issues.append(
            "compound_hypothesis"
        )

    if decision.unsupported_numeric_claims:
        issues.append(
            "unsupported_numeric_claims"
        )

    # Remove duplicates while preserving order.
    issues = list(dict.fromkeys(issues))

    return MechanismCritiqueReport(
        hypothesis_id=hypothesis.hypothesis_id,
        passed=len(issues) == 0,
        issues=issues,
        unsupported_numeric_claims=(
            decision.unsupported_numeric_claims
        ),
        rationale=decision.rationale,
    )