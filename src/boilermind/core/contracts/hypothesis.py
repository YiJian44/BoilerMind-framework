from pydantic import Field, model_validator

from boilermind.core.enums import (
    HypothesisStatus,
    MechanismSupportType,
)

from .base import ContractModel
from .experiment_memory import HypothesisProvenance


class MechanismStep(ContractModel):
    step: int = Field(ge=1)

    statement: str = Field(min_length=1)

    support_type: MechanismSupportType

    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_support(self):
        if (
            self.support_type
            == MechanismSupportType.VERIFIED_EVIDENCE
            and not self.evidence_ids
        ):
            raise ValueError(
                "verified_evidence mechanism step "
                "requires evidence_ids"
            )

        return self


class ScientificHypothesis(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    problem_id: str = Field(min_length=1)

    title: str = Field(min_length=1)

    research_significance: str = Field(min_length=1)

    hypothesis: str = Field(min_length=1)

    mechanism_chain: str = Field(min_length=1)

    mechanism_steps: list[MechanismStep] = Field(
        min_length=1
    )

    related_variables: list[str] = Field(
        default_factory=list
    )

    applicability_conditions: list[str] = Field(
        default_factory=list
    )

    verification_intent: str = Field(min_length=1)

    expected_observation: str = Field(min_length=1)

    confirmation_criteria: list[str] = Field(
        min_length=1
    )

    falsification_criteria: list[str] = Field(
        min_length=1
    )

    evidence_gaps: list[str] = Field(
        default_factory=list
    )

    assumptions: list[str] = Field(
        default_factory=list
    )

    counter_mechanisms: list[str] = Field(
        default_factory=list
    )

    novelty_axis: str = Field(min_length=1)

    evidence_bundle_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    status: HypothesisStatus = (
        HypothesisStatus.GENERATED
    )

    provenance: HypothesisProvenance | None = None


class HypothesisQualityReport(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    passed: bool

    issues: list[str] = Field(
        default_factory=list
    )

    traceable_step_count: int = Field(ge=0)

    total_step_count: int = Field(ge=1)

    evidence_coverage_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )


class MechanismCritiqueDecision(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    causal_chain_complete: bool

    physical_consistency: bool

    temporal_consistency: bool

    scope_consistency: bool

    single_testable_claim: bool

    unsupported_numeric_claims: list[str] = Field(
        default_factory=list
    )

    issues: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(min_length=1)


class MechanismCritiqueReport(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    passed: bool

    issues: list[str] = Field(
        default_factory=list
    )

    unsupported_numeric_claims: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(min_length=1)


class HypothesisAdmissionReport(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    passed: bool

    evidence_quality_passed: bool

    mechanism_critic_passed: bool

    issues: list[str] = Field(
        default_factory=list
    )
