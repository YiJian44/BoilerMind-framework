from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .base import ContractModel


class EvidenceTier(StrEnum):
    AUDITED_CONFIRMATORY = "AUDITED_CONFIRMATORY"
    AUDITED_EXPLORATORY = "AUDITED_EXPLORATORY"
    LEGACY_INFORMATIVE = "LEGACY_INFORMATIVE"
    ENGINEERING_FAILURE = "ENGINEERING_FAILURE"
    PLANNED_NOT_EXECUTED = "PLANNED_NOT_EXECUTED"


class ObservationType(StrEnum):
    SUPPORTED = "SUPPORTED"
    FALSIFIED = "FALSIFIED"
    PARTIAL = "PARTIAL"
    BOUNDARY_CONDITION = "BOUNDARY_CONDITION"
    CONTRADICTION = "CONTRADICTION"
    NULL_RESULT = "NULL_RESULT"
    ENGINEERING_FAILURE = "ENGINEERING_FAILURE"
    DATA_QUALITY_WARNING = "DATA_QUALITY_WARNING"


class ComparisonLevel(StrEnum):
    DIRECTLY_COMPARABLE = "DIRECTLY_COMPARABLE"
    CONDITIONALLY_COMPARABLE = "CONDITIONALLY_COMPARABLE"
    TRANSFER_ONLY = "TRANSFER_ONLY"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNKNOWN = "UNKNOWN"


class HypothesisTrigger(StrEnum):
    HISTORICAL_EXPERIMENT = "HISTORICAL_EXPERIMENT"
    CURRENT_DATA_OBSERVATION = "CURRENT_DATA_OBSERVATION"
    CONTRADICTORY_RESULTS = "CONTRADICTORY_RESULTS"
    LITERATURE_INSPIRATION = "LITERATURE_INSPIRATION"
    HUMAN_PROPOSAL = "HUMAN_PROPOSAL"
    CAPABILITY_EXPANSION = "CAPABILITY_EXPANSION"


class ApprovalStatus(StrEnum):
    PENDING_HUMAN_APPROVAL = "PENDING_HUMAN_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ExperimentScopeSignature(ContractModel):
    target_variable: str | None = None
    target_definition: str | None = None
    target_unit: str | None = None
    prediction_mode: str | None = None
    thermodynamic_standard: str | None = None
    dataset_id: str | None = None
    dataset_sha256: str | None = None
    feature_set_id: str | None = None
    feature_count: int | None = Field(default=None, ge=1)
    window_steps: int | None = Field(default=None, ge=0)
    prediction_horizon_steps: int | None = Field(default=None, ge=0)
    sampling_interval_seconds: int | None = Field(default=None, ge=1)
    split_policy: str | None = None
    split_ratios: list[float] = Field(default_factory=list)
    regime_definition: str | None = None
    metrics: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    protocol_status: str | None = None


class HistoricalExperimentRecord(ContractModel):
    schema_version: str = "boilermind.historical_experiment.v1"
    experiment_id: str = Field(min_length=1)
    series_id: str = Field(min_length=1)
    parent_experiment_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str | None = None
    run_date: str | None = None
    source_type: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_locator: str = Field(min_length=1)
    scope: ExperimentScopeSignature = Field(default_factory=ExperimentScopeSignature)
    random_seeds: list[int] = Field(default_factory=list)
    protocol_path: str | None = None
    candidate_models: list[str] = Field(default_factory=list)
    selection_scope: str | None = None
    locked_test_used_for_selection: bool | None = None
    confirmation_criteria: list[str] = Field(default_factory=list)
    falsification_criteria: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    regime_metrics: dict[str, Any] = Field(default_factory=dict)
    confidence_intervals: dict[str, Any] = Field(default_factory=dict)
    win_rates: dict[str, Any] = Field(default_factory=dict)
    verdict: str | None = None
    verdict_scope: list[str] = Field(default_factory=list)
    evidence_tier: EvidenceTier
    audit_status: str = "NOT_AUDITED"
    known_issues: list[str] = Field(default_factory=list)
    corrections: list[dict[str, Any]] = Field(default_factory=list)
    reproducibility_status: str = "UNKNOWN"
    artifact_paths: list[str] = Field(default_factory=list)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    raw_context: str = ""
    raw_hypothesis: str = ""
    raw_protocol: str = ""
    raw_result: str = ""
    raw_limitations: str = ""
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importer_version: str = "1.0.0"

    @property
    def scientific_synthesis_eligible(self) -> bool:
        if self.evidence_tier == EvidenceTier.AUDITED_CONFIRMATORY:
            return self.audit_status == "PASSED"
        return self.evidence_tier in {
            EvidenceTier.AUDITED_EXPLORATORY,
            EvidenceTier.LEGACY_INFORMATIVE,
        }


class ExperimentObservation(ContractModel):
    schema_version: str = "boilermind.experiment_observation.v1"
    observation_id: str = Field(min_length=1)
    source_experiment_ids: list[str] = Field(min_length=1)
    observation_type: ObservationType
    claim: str = Field(min_length=1)
    scope_signature: ExperimentScopeSignature = Field(default_factory=ExperimentScopeSignature)
    comparison_signature: str = ""
    supporting_metrics: dict[str, Any] = Field(default_factory=dict)
    counter_evidence: list[str] = Field(default_factory=list)
    confidence_level: float = Field(ge=0.0, le=1.0)
    reuse_policy: str = Field(min_length=1)
    invalid_for_scientific_synthesis: bool = False
    derived_by: str = "deterministic_observation_extractor"
    derivation_version: str = "1.0.0"


class ExperimentSeries(ContractModel):
    series_id: str = Field(min_length=1)
    experiment_ids: list[str] = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    evidence_tier: EvidenceTier


class ExperimentComparison(ContractModel):
    left_experiment_id: str = Field(min_length=1)
    right_experiment_id: str = Field(min_length=1)
    level: ComparisonLevel
    matched_fields: list[str] = Field(default_factory=list)
    mismatched_fields: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class ExperimentMemoryHit(ContractModel):
    experiment_id: str = Field(min_length=1)
    observation_ids: list[str] = Field(default_factory=list)
    comparison_level: ComparisonLevel
    retrieval_score: float = Field(ge=0.0, le=1.0)
    retrieval_reasons: list[str] = Field(default_factory=list)


class ExperimentMemoryBundle(ContractModel):
    problem_id: str = Field(min_length=1)
    directly_comparable: list[ExperimentMemoryHit] = Field(default_factory=list)
    conditionally_comparable: list[ExperimentMemoryHit] = Field(default_factory=list)
    transfer_only: list[ExperimentMemoryHit] = Field(default_factory=list)
    supported_observations: list[ExperimentObservation] = Field(default_factory=list)
    falsified_observations: list[ExperimentObservation] = Field(default_factory=list)
    contradictions: list[ExperimentObservation] = Field(default_factory=list)
    engineering_failures: list[ExperimentObservation] = Field(default_factory=list)
    completed_experiment_ids: list[str] = Field(default_factory=list)
    planned_hypothesis_ids: list[str] = Field(default_factory=list)
    retrieval_notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CurrentObservationBundle(ContractModel):
    problem_id: str = Field(min_length=1)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Opportunity(ContractModel):
    opportunity_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    trigger_types: list[HypothesisTrigger] = Field(min_length=1)
    source_observation_ids: list[str] = Field(default_factory=list)
    source_experiment_ids: list[str] = Field(default_factory=list)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    currently_executable: bool
    missing_capabilities: list[str] = Field(default_factory=list)
    do_not_repeat_experiment_ids: list[str] = Field(default_factory=list)


class OpportunityMap(ContractModel):
    problem_id: str = Field(min_length=1)
    opportunities: list[Opportunity] = Field(default_factory=list)
    stop_reasons: list[str] = Field(default_factory=list)
    source_quota: dict[str, int] = Field(default_factory=lambda: {
        "experiment_or_data": 7,
        "boundary_or_capability": 2,
        "literature_or_human": 1,
    })
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HypothesisProvenance(ContractModel):
    trigger_types: list[HypothesisTrigger] = Field(min_length=1)
    source_observation_ids: list[str] = Field(default_factory=list)
    source_experiment_ids: list[str] = Field(default_factory=list)
    source_literature_ids: list[str] = Field(default_factory=list)
    human_proposal_ids: list[str] = Field(default_factory=list)
    opportunity_id: str | None = None
    scope_signature: ExperimentScopeSignature = Field(default_factory=ExperimentScopeSignature)
    expected_information_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_of: str | None = None
    supersedes_hypothesis_ids: list[str] = Field(default_factory=list)
    capability_match_status: str = "UNKNOWN"


class ModelCapabilityPerformance(ContractModel):
    model_id: str = Field(min_length=1)
    run_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    convergence_failure_count: int = Field(ge=0)
    runtime_seconds: list[float] = Field(default_factory=list)
    common_failure_reasons: list[str] = Field(default_factory=list)
    last_verified_at: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class EmpiricalCapabilityProfile(ContractModel):
    schema_version: str = "boilermind.empirical_capability.v1"
    models: list[ModelCapabilityPerformance] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NextRoundCandidate(ContractModel):
    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_observation_ids: list[str] = Field(default_factory=list)
    source_experiment_ids: list[str] = Field(default_factory=list)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    required_capabilities: list[str] = Field(default_factory=list)
    estimated_cost: str = "UNKNOWN"
    known_risks: list[str] = Field(default_factory=list)


class NextRoundProposalBundle(ContractModel):
    proposal_id: str = Field(min_length=1)
    source_experiment_id: str = Field(min_length=1)
    new_observation_ids: list[str] = Field(default_factory=list)
    candidates: list[NextRoundCandidate] = Field(default_factory=list)
    recommended_candidate_id: str | None = None
    stop_reasons: list[str] = Field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_HUMAN_APPROVAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def require_human_approval(self):
        if self.approval_status == ApprovalStatus.APPROVED and not self.candidates:
            raise ValueError("approved_next_round_requires_candidate")
        return self


class LiteratureRelation(ContractModel):
    document_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    relationship: str = Field(pattern=r"^(SUPPORTING|CONTRADICTING|METHOD_RELATED|DOMAIN_BACKGROUND|TRANSFER_REFERENCE|LIMITATION_REFERENCE)$")
    page_number: int | None = Field(default=None, ge=1)
    chunk_id: str | None = None
    excerpt: str = ""
    applicability: str = "UNKNOWN"
    metadata_verified: bool = False
    formatted_citation: str | None = None
