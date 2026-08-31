from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from boilermind.core.contracts import (
    EvidenceCandidate,
    EvidenceVerificationDecision,
    VerifiedEvidence,
)
from boilermind.evidence.qwen_semantic_judge import (
    SemanticEvidenceAssessment,
)
from boilermind.evidence.traceability_verifier import (
    TraceabilityResult,
)
from boilermind.evidence.verifier import (
    EvidenceVerificationError,
    promote_candidate,
)
from boilermind.evidence.content_policy import is_reference_list_only
from boilermind.core.enums import ApplicabilityLevel, ClaimSupport


@dataclass(frozen=True)
class EvidenceRejection:
    evidence_id: str
    reason: str


@dataclass(frozen=True)
class VerificationPipelineResult:
    decisions: tuple[EvidenceVerificationDecision, ...]
    verified: tuple[VerifiedEvidence, ...]
    rejected: tuple[EvidenceRejection, ...]


def build_verification_decision(
    candidate: EvidenceCandidate,
    traceability: TraceabilityResult,
    semantic: SemanticEvidenceAssessment,
) -> EvidenceVerificationDecision:
    """
    Combine deterministic provenance verification
    and Qwen semantic verification.

    Neither layer may override failure in the other.
    """

    if semantic.evidence_id != candidate.evidence_id:
        raise ValueError(
            "Semantic assessment evidence_id does not "
            "match candidate evidence_id."
        )

    content_hash = sha256(
        candidate.text.encode("utf-8")
    ).hexdigest()

    reference_list_only = is_reference_list_only(candidate.text)
    hypothesis_inspiration_eligible = (
        traceability.verified
        and semantic.semantic_verified
        and semantic.core_claim_eligible
        and semantic.claim_support in {ClaimSupport.DIRECT, ClaimSupport.PARTIAL}
        and semantic.applicability in {ApplicabilityLevel.HIGH, ApplicabilityLevel.MEDIUM}
        and not reference_list_only
    )
    formal_claim_support_eligible = (
        traceability.verified
        and semantic.semantic_verified
        and semantic.claim_support == ClaimSupport.DIRECT
        and semantic.applicability == ApplicabilityLevel.HIGH
        and candidate.human_citation_approved
        and candidate.citation_eligibility == "FORMALLY_CITABLE"
        and not reference_list_only
    )

    rationale = (
        "Traceability: "
        f"{traceability.rationale} "
        "Semantic: "
        f"{semantic.verification_rationale}"
    )

    return EvidenceVerificationDecision(
        evidence_id=candidate.evidence_id,
        citation_verified=traceability.verified,
        semantic_verified=semantic.semantic_verified,
        claim_support=semantic.claim_support,
        applicability=semantic.applicability,
        core_claim_eligible=hypothesis_inspiration_eligible,
        hypothesis_inspiration_eligible=hypothesis_inspiration_eligible,
        formal_claim_support_eligible=formal_claim_support_eligible,
        verification_rationale=rationale,
        source_hash=traceability.source_hash,
        content_hash=content_hash,
    )


def verify_from_assessments(
    candidates: list[EvidenceCandidate],
    traceability_by_id: Mapping[
        str,
        TraceabilityResult,
    ],
    semantic_by_id: Mapping[
        str,
        SemanticEvidenceAssessment,
    ],
) -> VerificationPipelineResult:
    """
    Fail closed if any candidate is missing either
    traceability or semantic verification.

    Promotion to VerifiedEvidence is delegated to the
    existing trusted verifier.
    """

    candidate_ids = [
        candidate.evidence_id
        for candidate in candidates
    ]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(
            "Duplicate candidate evidence IDs."
        )

    decisions: list[
        EvidenceVerificationDecision
    ] = []

    verified: list[
        VerifiedEvidence
    ] = []

    rejected: list[
        EvidenceRejection
    ] = []

    for candidate in candidates:
        evidence_id = candidate.evidence_id

        if evidence_id not in traceability_by_id:
            raise ValueError(
                "Missing traceability result for "
                f"{evidence_id}."
            )

        if evidence_id not in semantic_by_id:
            raise ValueError(
                "Missing semantic assessment for "
                f"{evidence_id}."
            )

        decision = build_verification_decision(
            candidate,
            traceability_by_id[evidence_id],
            semantic_by_id[evidence_id],
        )

        decisions.append(decision)

        try:
            item = promote_candidate(
                candidate,
                decision,
            )
        except EvidenceVerificationError as exc:
            rejected.append(
                EvidenceRejection(
                    evidence_id=evidence_id,
                    reason=str(exc),
                )
            )
        else:
            verified.append(item)

    return VerificationPipelineResult(
        decisions=tuple(decisions),
        verified=tuple(verified),
        rejected=tuple(rejected),
    )
