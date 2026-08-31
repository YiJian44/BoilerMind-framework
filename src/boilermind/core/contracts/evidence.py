from datetime import datetime

from pydantic import Field, model_validator

from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
    EvidenceStage,
)

from .base import ContractModel


class EvidenceCandidate(ContractModel):
    evidence_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)

    stage: EvidenceStage = EvidenceStage.CANDIDATE

    source_type: str = Field(min_length=1)
    title: str = Field(min_length=1)

    source_url: str | None = None
    citation: str | None = None

    text: str = Field(min_length=1)

    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    retrieved_at: datetime

    # Exact provenance for literature evidence.
    document_id: str | None = None
    chunk_id: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    chunk_index: int | None = Field(
        default=None,
        ge=0,
    )

    corpus_level: str | None = None
    source_file: str | None = None
    metadata_status: str | None = None

    # Bibliographic identity and formal-citation state are
    # deliberately separate from source traceability. A paper
    # may be useful for retrieval while remaining ineligible for
    # the final reference list.
    identity_status: str | None = None
    citation_candidate_eligibility: str | None = None
    human_citation_approved: bool = False
    citation_eligibility: str | None = None
    formatted_citation: str | None = None

    document_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class EvidenceVerificationDecision(ContractModel):
    evidence_id: str = Field(min_length=1)

    citation_verified: bool
    semantic_verified: bool

    claim_support: ClaimSupport
    applicability: ApplicabilityLevel

    core_claim_eligible: bool
    # Backward-compatible core_claim_eligible remains the hypothesis
    # inspiration gate. Formal report support is deliberately stricter.
    hypothesis_inspiration_eligible: bool = False
    formal_claim_support_eligible: bool = False

    verification_rationale: str = Field(
        min_length=1
    )

    source_hash: str | None = None
    content_hash: str | None = None


class VerifiedEvidence(ContractModel):
    evidence_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)

    stage: EvidenceStage = EvidenceStage.VERIFIED

    source_type: str = Field(min_length=1)
    title: str = Field(min_length=1)

    source_url: str | None = None
    citation: str | None = None

    text: str = Field(min_length=1)

    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    # Preserve exact source provenance after verification.
    document_id: str | None = None
    chunk_id: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    chunk_index: int | None = Field(
        default=None,
        ge=0,
    )

    corpus_level: str | None = None
    source_file: str | None = None
    metadata_status: str | None = None
    identity_status: str | None = None
    citation_candidate_eligibility: str | None = None
    human_citation_approved: bool = False
    citation_eligibility: str | None = None
    formatted_citation: str | None = None

    document_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    citation_verified: bool
    semantic_verified: bool

    claim_support: ClaimSupport
    applicability: ApplicabilityLevel

    core_claim_eligible: bool
    hypothesis_inspiration_eligible: bool = False
    formal_claim_support_eligible: bool = False

    verification_rationale: str = Field(
        min_length=1
    )

    source_hash: str | None = None
    content_hash: str | None = None

    @model_validator(mode="after")
    def validate_verified_evidence(self):
        # Data observations are verified by a frozen dataset plus an audited,
        # reproducible execution pipeline. They are deliberately NOT external
        # papers: no citation/semantic verification applies, and they must
        # never be claimed as formally citable literature.
        if self.source_type == "DATA_OBSERVATION":
            if self.formal_claim_support_eligible:
                raise ValueError(
                    "DATA_OBSERVATION evidence cannot be "
                    "formal_claim_support_eligible"
                )
            return self

        if not self.citation_verified:
            raise ValueError(
                "VerifiedEvidence requires "
                "citation_verified=True"
            )

        if not self.semantic_verified:
            raise ValueError(
                "VerifiedEvidence requires "
                "semantic_verified=True"
            )

        if (
            self.core_claim_eligible
            and self.claim_support
            not in {
                ClaimSupport.DIRECT,
                ClaimSupport.PARTIAL,
            }
        ):
            raise ValueError(
                "Core evidence requires direct "
                "or partial support"
            )

        # Older trusted callers only populated core_claim_eligible.
        # Treat it as the legacy name for hypothesis inspiration.
        if self.core_claim_eligible and not self.hypothesis_inspiration_eligible:
            object.__setattr__(self, "hypothesis_inspiration_eligible", True)
        elif self.hypothesis_inspiration_eligible and not self.core_claim_eligible:
            raise ValueError(
                "Hypothesis inspiration eligibility requires the legacy core gate"
            )

        if self.formal_claim_support_eligible and (
            self.claim_support != ClaimSupport.DIRECT
            or self.applicability != ApplicabilityLevel.HIGH
            or not self.human_citation_approved
            or self.citation_eligibility != "FORMALLY_CITABLE"
        ):
            raise ValueError(
                "Formal claim support requires direct/high support "
                "and an approved formally citable source"
            )

        return self


class EvidenceBundle(ContractModel):
    bundle_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)

    evidence: list[VerifiedEvidence] = Field(
        min_length=1
    )

    created_at: datetime

    sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
