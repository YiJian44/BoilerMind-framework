from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from boilermind.core.contracts import (
    EvidenceBundle,
    MechanismStep,
    VerifiedEvidence,
)

from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
    MechanismSupportType,
)


def test_unverified_evidence_cannot_be_core_claim():
    with pytest.raises(ValidationError):
        VerifiedEvidence(
            evidence_id="E001",
            problem_id="P001",
            source_type="local_rag",
            title="Example",
            text="Example evidence text",
            retrieval_score=0.8,
            citation_verified=False,
            semantic_verified=True,
            claim_support=ClaimSupport.DIRECT,
            applicability=ApplicabilityLevel.HIGH,
            core_claim_eligible=True,
            verification_rationale="Test",
        )


def test_verified_mechanism_step_requires_evidence_id():
    with pytest.raises(ValidationError):
        MechanismStep(
            step=1,
            statement="Fuel change affects heat release.",
            support_type=MechanismSupportType.VERIFIED_EVIDENCE,
            evidence_ids=[],
        )


def test_evidence_bundle_requires_real_sha256():
    evidence = VerifiedEvidence(
        evidence_id="E001",
        problem_id="P001",
        source_type="local_rag",
        title="Example",
        text="Example evidence text",
        retrieval_score=0.8,
        citation_verified=True,
        semantic_verified=True,
        claim_support=ClaimSupport.DIRECT,
        applicability=ApplicabilityLevel.HIGH,
        core_claim_eligible=True,
        verification_rationale="Verified for test.",
    )

    with pytest.raises(ValidationError):
        EvidenceBundle(
            bundle_id="B001",
            problem_id="P001",
            evidence=[evidence],
            created_at=datetime.now(timezone.utc),
            sha256="fake",
        )