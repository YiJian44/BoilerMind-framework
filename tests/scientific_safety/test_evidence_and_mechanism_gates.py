from datetime import datetime, timezone

import pytest

from boilermind.core.contracts import (
    EvidenceBundle,
    EvidenceCandidate,
    EvidenceVerificationDecision,
    MechanismCritiqueDecision,
    MechanismStep,
    ScientificHypothesis,
)

from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
    MechanismSupportType,
)

from boilermind.evidence.verifier import (
    EvidenceVerificationError,
    promote_candidate,
)

from boilermind.hypothesis.admission import (
    admit_hypothesis,
)


BUNDLE_HASH = "b" * 64


def make_candidate(
    source_type="local_rag",
):
    return EvidenceCandidate(
        evidence_id="E001",
        problem_id="P001",
        source_type=source_type,
        title="Boiler evidence",
        text="Verified boiler evidence text.",
        retrieval_score=0.9,
        retrieved_at=datetime.now(timezone.utc),
    )


def make_decision():
    return EvidenceVerificationDecision(
        evidence_id="E001",
        citation_verified=True,
        semantic_verified=True,
        claim_support=ClaimSupport.DIRECT,
        applicability=ApplicabilityLevel.HIGH,
        core_claim_eligible=True,
        verification_rationale="Verified for test.",
    )


def make_hypothesis():
    return ScientificHypothesis(
        hypothesis_id="H001",
        problem_id="P001",
        title="Dynamic lag hypothesis",
        research_significance="Research significance",
        hypothesis="Dynamic lag increases prediction error.",
        mechanism_chain="A -> B -> C",
        mechanism_steps=[
            MechanismStep(
                step=1,
                statement="Verified physical relation",
                support_type=(
                    MechanismSupportType.VERIFIED_EVIDENCE
                ),
                evidence_ids=["E001"],
            ),
            MechanismStep(
                step=2,
                statement="Effect to be tested",
                support_type=(
                    MechanismSupportType.HYPOTHESIS_INFERENCE
                ),
                evidence_ids=[],
            ),
        ],
        related_variables=[
            "fuel",
            "feedwater",
        ],
        applicability_conditions=[
            "deep peak regulation"
        ],
        verification_intent=(
            "Compare dynamic compensation "
            "against baseline."
        ),
        expected_observation=(
            "Dynamic model has lower error."
        ),
        confirmation_criteria=[
            "Predefined target is achieved."
        ],
        falsification_criteria=[
            "Predefined target is not achieved."
        ],
        evidence_gaps=[
            "Plant validation required."
        ],
        assumptions=[
            "Sensors are valid."
        ],
        counter_mechanisms=[
            "Thermal storage may dominate."
        ],
        novelty_axis="Dynamic lag mechanism",
        evidence_bundle_sha256=BUNDLE_HASH,
    )


def make_bundle():
    evidence = promote_candidate(
        make_candidate(),
        make_decision(),
    )

    return EvidenceBundle(
        bundle_id="B001",
        problem_id="P001",
        evidence=[evidence],
        created_at=datetime.now(timezone.utc),
        sha256=BUNDLE_HASH,
    )


def make_good_mechanism_decision():
    return MechanismCritiqueDecision(
        hypothesis_id="H001",
        causal_chain_complete=True,
        physical_consistency=True,
        temporal_consistency=True,
        scope_consistency=True,
        single_testable_claim=True,
        unsupported_numeric_claims=[],
        issues=[],
        rationale="Mechanism is testable and consistent.",
    )


def test_fixture_cannot_become_core_evidence():
    with pytest.raises(EvidenceVerificationError):
        promote_candidate(
            make_candidate("artifact_replay"),
            make_decision(),
        )


def test_semantic_failure_cannot_be_promoted():
    decision = make_decision().model_copy(
        update={
            "semantic_verified": False,
        }
    )

    with pytest.raises(EvidenceVerificationError):
        promote_candidate(
            make_candidate(),
            decision,
        )


def test_valid_hypothesis_is_admitted():
    hypothesis, report = admit_hypothesis(
        make_hypothesis(),
        make_bundle(),
        make_good_mechanism_decision(),
    )

    assert report.passed is True
    assert hypothesis.status.value == "qualified"


def test_physical_inconsistency_is_rejected():
    bad_decision = (
        make_good_mechanism_decision().model_copy(
            update={
                "physical_consistency": False,
                "rationale": "Physical inconsistency detected.",
            }
        )
    )

    hypothesis, report = admit_hypothesis(
        make_hypothesis(),
        make_bundle(),
        bad_decision,
    )

    assert report.passed is False
    assert hypothesis.status.value == "rejected"
    assert "physical_inconsistency" in report.issues


def test_unsupported_numeric_claim_is_rejected():
    bad_decision = (
        make_good_mechanism_decision().model_copy(
            update={
                "unsupported_numeric_claims": [
                    "30-120 seconds"
                ],
                "rationale": (
                    "Numeric range has no evidence."
                ),
            }
        )
    )

    hypothesis, report = admit_hypothesis(
        make_hypothesis(),
        make_bundle(),
        bad_decision,
    )

    assert report.passed is False
    assert "unsupported_numeric_claims" in report.issues