from boilermind.core.contracts import (
    EvidenceCandidate,
    EvidenceVerificationDecision,
    VerifiedEvidence,
)

from boilermind.core.enums import ClaimSupport
from boilermind.evidence.content_policy import is_reference_list_only


class EvidenceVerificationError(ValueError):
    pass


_BLOCKED_CORE_SOURCE_TYPES = {
    "test_fixture",
    "fixture",
    "artifact_replay",
    "mock",
    "demo_reference",
}


def promote_candidate(
    candidate: EvidenceCandidate,
    decision: EvidenceVerificationDecision,
) -> VerifiedEvidence:
    if candidate.evidence_id != decision.evidence_id:
        raise EvidenceVerificationError(
            "Evidence decision ID does not match candidate ID."
        )

    if not decision.citation_verified:
        raise EvidenceVerificationError(
            "Citation verification failed."
        )

    if not decision.semantic_verified:
        raise EvidenceVerificationError(
            "Semantic verification failed."
        )

    if decision.claim_support in {ClaimSupport.IRRELEVANT, ClaimSupport.UNKNOWN}:
        raise EvidenceVerificationError(
            "Irrelevant or unknown evidence cannot become verified scientific evidence."
        )

    if is_reference_list_only(candidate.text):
        raise EvidenceVerificationError(
            "Reference-list-only content cannot become verified scientific evidence."
        )

    source_type = candidate.source_type.strip().lower()

    if (
        source_type in _BLOCKED_CORE_SOURCE_TYPES
        and decision.core_claim_eligible
    ):
        raise EvidenceVerificationError(
            f"Source type '{candidate.source_type}' "
            "cannot become core scientific evidence."
        )

    if (
        decision.core_claim_eligible
        and decision.claim_support
        not in {
            ClaimSupport.DIRECT,
            ClaimSupport.PARTIAL,
        }
    ):
        raise EvidenceVerificationError(
            "Core scientific evidence must directly "
            "or partially support the claim."
        )

    return VerifiedEvidence(
        evidence_id=candidate.evidence_id,
        problem_id=candidate.problem_id,
        source_type=candidate.source_type,
        title=candidate.title,
        source_url=candidate.source_url,
        citation=candidate.citation,
        text=candidate.text,
        retrieval_score=candidate.retrieval_score,
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        page_number=candidate.page_number,
        chunk_index=candidate.chunk_index,
        corpus_level=candidate.corpus_level,
        source_file=candidate.source_file,
        metadata_status=candidate.metadata_status,
        identity_status=candidate.identity_status,
        citation_candidate_eligibility=candidate.citation_candidate_eligibility,
        human_citation_approved=candidate.human_citation_approved,
        citation_eligibility=candidate.citation_eligibility,
        formatted_citation=candidate.formatted_citation,
        document_sha256=candidate.document_sha256,
        citation_verified=decision.citation_verified,
        semantic_verified=decision.semantic_verified,
        claim_support=decision.claim_support,
        applicability=decision.applicability,
        core_claim_eligible=decision.core_claim_eligible,
        hypothesis_inspiration_eligible=(
            decision.hypothesis_inspiration_eligible
            or decision.core_claim_eligible
        ),
        formal_claim_support_eligible=decision.formal_claim_support_eligible,
        verification_rationale=(
            decision.verification_rationale
        ),
        source_hash=decision.source_hash,
        content_hash=decision.content_hash,
    )


def verify_candidates(
    candidates: list[EvidenceCandidate],
    decisions: dict[
        str,
        EvidenceVerificationDecision,
    ],
) -> tuple[
    list[VerifiedEvidence],
    dict[str, str],
]:
    accepted: list[VerifiedEvidence] = []
    rejected: dict[str, str] = {}

    for candidate in candidates:
        decision = decisions.get(candidate.evidence_id)

        if decision is None:
            rejected[candidate.evidence_id] = (
                "missing_verification_decision"
            )
            continue

        try:
            accepted.append(
                promote_candidate(
                    candidate,
                    decision,
                )
            )
        except EvidenceVerificationError as exc:
            rejected[candidate.evidence_id] = str(exc)

    return accepted, rejected
