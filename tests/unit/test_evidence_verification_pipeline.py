from datetime import datetime, timezone

import pytest

from boilermind.core.contracts import (
    EvidenceCandidate,
)
from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
)
from boilermind.evidence.bundle_freezer import (
    EvidenceBundleFreezeError,
    freeze_evidence_bundle,
)
from boilermind.evidence.qwen_semantic_judge import (
    SemanticEvidenceAssessment,
)
from boilermind.evidence.traceability_verifier import (
    TraceabilityResult,
)
from boilermind.evidence.verification_pipeline import (
    verify_from_assessments,
)


def _candidate(
    evidence_id: str,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        problem_id="P-VERIFY-001",
        source_type="local_literature",
        title="Boiler prediction paper",
        citation="Test citation",
        text=(
            "This study investigates prediction "
            "performance in an industrial boiler."
        ),
        retrieval_score=0.9,
        retrieved_at=datetime.now(
            timezone.utc
        ),
        human_citation_approved=True,
        citation_eligibility="FORMALLY_CITABLE",
    )


def _traceable() -> TraceabilityResult:
    return TraceabilityResult(
        verified=True,
        rationale=(
            "Source provenance is traceable."
        ),
        source_hash="a" * 64,
    )


def _relevant(
    evidence_id: str,
) -> SemanticEvidenceAssessment:
    return SemanticEvidenceAssessment(
        evidence_id=evidence_id,
        semantic_verified=True,
        claim_support=ClaimSupport.DIRECT,
        applicability=ApplicabilityLevel.HIGH,
        core_claim_eligible=True,
        verification_rationale=(
            "Directly relevant to the research problem."
        ),
    )


def test_verified_evidence_requires_both_layers():
    candidate = _candidate("E-001")

    result = verify_from_assessments(
        [candidate],
        {
            "E-001": _traceable(),
        },
        {
            "E-001": _relevant(
                "E-001"
            ),
        },
    )

    assert len(result.decisions) == 1
    assert len(result.verified) == 1
    assert len(result.rejected) == 0
    assert result.verified[0].hypothesis_inspiration_eligible is True
    assert result.verified[0].formal_claim_support_eligible is True


def test_semantically_irrelevant_candidate_is_rejected():
    candidate = _candidate("E-002")

    semantic = SemanticEvidenceAssessment(
        evidence_id="E-002",
        semantic_verified=False,
        claim_support=ClaimSupport.IRRELEVANT,
        applicability=ApplicabilityLevel.LOW,
        core_claim_eligible=False,
        verification_rationale=(
            "Keyword overlap only."
        ),
    )

    result = verify_from_assessments(
        [candidate],
        {
            "E-002": _traceable(),
        },
        {
            "E-002": semantic,
        },
    )

    assert len(result.verified) == 0
    assert len(result.rejected) == 1


def test_inconsistent_irrelevant_but_semantic_true_is_rejected():
    candidate = _candidate("E-IRRELEVANT")
    semantic = SemanticEvidenceAssessment(
        evidence_id=candidate.evidence_id,
        semantic_verified=True,
        claim_support=ClaimSupport.IRRELEVANT,
        applicability=ApplicabilityLevel.LOW,
        core_claim_eligible=False,
        verification_rationale="No scientific support.",
    )
    result = verify_from_assessments(
        [candidate], {candidate.evidence_id: _traceable()},
        {candidate.evidence_id: semantic},
    )
    assert not result.verified
    assert "Irrelevant or unknown" in result.rejected[0].reason


def test_reference_list_only_cannot_enter_verified_bundle():
    candidate = _candidate("E-REFS").model_copy(update={
        "text": "REFERENCES\n[1] A. Author. Journal, 2020.\n[2] B. Author. Proceedings, 2021.",
    })
    result = verify_from_assessments(
        [candidate], {candidate.evidence_id: _traceable()},
        {candidate.evidence_id: _relevant(candidate.evidence_id)},
    )
    assert not result.verified
    assert result.decisions[0].core_claim_eligible is False
    assert result.decisions[0].formal_claim_support_eligible is False
    assert "Reference-list-only" in result.rejected[0].reason


def test_partial_support_can_inspire_but_not_support_formal_claim():
    candidate = _candidate("E-PARTIAL")
    semantic = SemanticEvidenceAssessment(
        evidence_id=candidate.evidence_id,
        semantic_verified=True,
        claim_support=ClaimSupport.PARTIAL,
        applicability=ApplicabilityLevel.MEDIUM,
        core_claim_eligible=True,
        verification_rationale="Transferable method only.",
    )
    result = verify_from_assessments(
        [candidate], {candidate.evidence_id: _traceable()},
        {candidate.evidence_id: semantic},
    )
    assert result.verified[0].hypothesis_inspiration_eligible is True
    assert result.verified[0].formal_claim_support_eligible is False


def test_missing_semantic_assessment_fails_closed():
    candidate = _candidate("E-003")

    with pytest.raises(
        ValueError,
        match="Missing semantic assessment",
    ):
        verify_from_assessments(
            [candidate],
            {
                "E-003": _traceable(),
            },
            {},
        )


def test_evidence_bundle_hash_is_deterministic():
    first = _candidate("E-004")
    second = _candidate("E-005")

    result = verify_from_assessments(
        [first, second],
        {
            "E-004": _traceable(),
            "E-005": _traceable(),
        },
        {
            "E-004": _relevant(
                "E-004"
            ),
            "E-005": _relevant(
                "E-005"
            ),
        },
    )

    evidence = list(result.verified)

    bundle_a = freeze_evidence_bundle(
        "P-VERIFY-001",
        evidence,
    )

    bundle_b = freeze_evidence_bundle(
        "P-VERIFY-001",
        list(reversed(evidence)),
    )

    assert bundle_a.sha256 == bundle_b.sha256
    assert bundle_a.bundle_id == bundle_b.bundle_id
    assert len(bundle_a.sha256) == 64


def test_empty_bundle_cannot_be_frozen():
    with pytest.raises(
        EvidenceBundleFreezeError,
        match="empty evidence bundle",
    ):
        freeze_evidence_bundle(
            "P-VERIFY-001",
            [],
        )
