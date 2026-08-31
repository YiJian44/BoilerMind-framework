"""Final, read-only scientific output assembled after experiment verdict."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from boilermind.core.enums import ScientificVerdict

from .base import ContractModel


class ScientificResearchPlanMetadata(ContractModel):
    schema_version: str | None
    report_id: str | None
    generated_at: datetime | None
    problem_id: str | None
    hypothesis_id: str | None
    plan_id: str | None
    experiment_id: str | None
    report_status: str | None
    run_id: str | None = None
    report_phase: str = "POST_EXPERIMENT"
    generator_version: str = "2.0.0"


class ProblemStatementSection(ContractModel):
    original_question: str | None
    research_object: str | None
    target_variable: str | None
    objective: str | None
    metrics: list[str] | None
    target_inference_reason: str | None
    operating_condition: str | None
    manipulated_variables: tuple[str, ...] | None
    observed_variables: tuple[str, ...] | None
    context_variables: tuple[str, ...] | None
    research_goal: str | None
    success_criteria: list[str] | None
    constraints: list[str] | None
    current_limitation: str | None = None
    research_gap: str | None = None
    limitation_evidence_ids: list[str] = Field(default_factory=list)
    scope_boundary: list[str] = Field(default_factory=list)


class RationaleSection(ContractModel):
    research_significance: str | None
    hypothesis_statement: str | None
    mechanism_chain: str | None
    mechanism_steps: list[dict[str, Any]] | None
    related_variables: list[str] | None
    applicability_conditions: list[str] | None
    verification_intent: str | None
    expected_observation: str | None
    assumptions: list[str] | None
    counter_mechanisms: list[str] | None
    evidence_gaps: list[str] | None
    novelty_axis: str | None
    evidence_bundle_sha256: str | None
    confirmation_criteria: tuple[str, ...] | None
    falsification_criteria: tuple[str, ...] | None
    innovation_point: str | None = None
    reasoning_chain: list[dict[str, Any]] = Field(default_factory=list)
    competing_hypotheses: list[dict[str, Any]] = Field(default_factory=list)


class TechnicalDetailsSection(ContractModel):
    experiment_type: str | None
    required_operations: list[str] | None
    window_steps: int | None
    prediction_horizon_steps: int | None
    sampling_interval_seconds: int | None
    random_seed: int | None
    execution_requirements: dict[str, Any] | None
    allowed_devices: list[str] | None
    reuse_checkpoint_models: list[str] | None
    technical_stack: list[dict[str, Any]] = Field(default_factory=list)
    preprocessing_policy: list[str] = Field(default_factory=list)
    leakage_prevention_policy: list[str] = Field(default_factory=list)
    reproducibility_controls: list[str] = Field(default_factory=list)


class DatasetSection(ContractModel):
    source: str | None
    dataset_id: str | None
    dataset_hash: str | None
    dataset_path: str | None
    target: str | None
    input_variables: list[str] | None
    train_split: str | None
    validation_split: str | None
    locked_test_split: str | None
    scaler_fit_scope: str | None
    chronological_split: bool | None
    sample_counts: dict[str, int] | None
    source_type: str = "LOCAL_INDUSTRIAL_DATA"
    source_compliance_status: str = "VERIFIED"
    source_usage_authorized: bool = True
    historical_data_scope: str | None = None
    target_definition_id: str | None = None
    target_formula_version: str | None = None
    target_unit: str | None = None
    proposed_additional_features: list[str] = Field(default_factory=list)
    collection_required: bool = False
    collection_description: str | None = None
    train_only_preprocessing: bool | None = None
    locked_test_used_for_selection: bool | None = None


class MethodsSection(ContractModel):
    objective: str | None
    experimental_design: str | None
    baseline_description: str | None
    intervention_description: str | None
    control: dict[str, Any] | None
    treatment: dict[str, Any] | None
    recommended_models: list[str] | None
    executable_models: list[str] | None
    candidate_models: list[str] | None
    reference_models: tuple[str, ...] | None
    model_selection_rationale: str | None
    model_substitution_reason: str | None
    primary_metric: str | None
    secondary_metrics: tuple[str, ...] | None
    locked_test_used_for_selection: bool | None
    execution_backend: str | None
    allow_partial_failure: bool | None
    max_runtime_per_model: float | None
    max_epochs: int | None
    confirmation_criteria: list[str] | None
    falsification_criteria: list[str] | None
    implementation_steps: list[dict[str, Any]] = Field(default_factory=list)
    hyperparameter_policy: str | None = None
    model_selection_policy: str | None = None
    validation_policy: str | None = None
    locked_test_policy: str | None = None


class ModelResultSnapshot(ContractModel):
    model_name: str
    fit_success: bool | None
    fit_converged: bool | None
    runtime_seconds: float | None
    model_configuration: dict[str, Any] | None = None
    validation_metrics: dict[str, float] | None
    locked_test_metrics: dict[str, float] | None
    warnings: tuple[str, ...] | None
    failure_reason: str | None
    sample_counts: dict[str, int] | None
    random_seed: int | None
    device: str | None
    artifact_provenance: dict[str, Any] | None


class ExperimentsSection(ContractModel):
    experiment_id: str | None
    status: str | None
    started_at: datetime | None
    completed_at: datetime | None
    model_results: list[ModelResultSnapshot] | None
    execution_notes: list[str] | None
    artifacts: list[str] | None
    baselines: list[dict[str, Any]] = Field(default_factory=list)
    metric_definitions: list[dict[str, Any]] = Field(default_factory=list)
    expected_observation: str | None = None


class ResultsSection(ContractModel):
    overall_metrics: dict[str, float] | None
    baseline_metrics: dict[str, float] | None
    candidate_locked_test_metrics: dict[str, dict[str, float]] | None
    control_metrics: dict[str, float] | None
    treatment_metrics: dict[str, float] | None
    metric_deltas: dict[str, float] | None
    result_status: str = "COMPLETED"
    feasibility_basis: str = "ACTUAL_EXECUTION"
    formula_derivation: list[str] = Field(default_factory=list)
    achieved_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    scientific_verdict: str | None = None
    verdict_rationale: str | None = None
    experiment_valid: bool | None = None
    audit_issues: list[str] = Field(default_factory=list)
    protocol_selected_model: str | None = None
    locked_test_best_model: str | None = None
    selection_interpretation: str | None = None
    model_comparison_rows: list[dict[str, Any]] = Field(default_factory=list)


class MetricsSection(ContractModel):
    planned_metrics: list[str] | None
    primary_metric: str | None
    secondary_metrics: list[str] | None
    validation_metrics_by_model: dict[str, dict[str, float]] | None
    locked_test_metrics_by_model: dict[str, dict[str, float]] | None
    baseline_metrics: dict[str, float] | None
    control_metrics: dict[str, float] | None
    treatment_metrics: dict[str, float] | None
    metric_deltas: dict[str, float] | None
    metric_unit: str | None = None


class ResearchTraceEntry(ContractModel):
    plan_id: str | None
    experiment_id: str | None
    status: str | None
    metrics: dict[str, Any] | None
    target_met: bool | None
    reason: str | None


class ScientificVerdictSnapshot(ContractModel):
    """Immutable copy of an existing ScientificResult; never recomputed here."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    verdict: ScientificVerdict | None
    rationale: str | None
    achieved_criteria: tuple[str, ...] | None
    failed_criteria: tuple[str, ...] | None
    source_hypothesis_id: str | None
    source_experiment_id: str | None


class ExperimentValiditySnapshot(ContractModel):
    """Immutable copy of existing result and audit validity fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    experiment_valid: bool | None
    execution_valid: bool | None
    dataset_frozen: bool | None
    leakage_check_passed: bool | None
    baseline_valid: bool | None
    metric_check_passed: bool | None
    issues: tuple[str, ...] | None
    validity_source: str | None


class ReferenceEntry(ContractModel):
    evidence_id: str | None
    title: str | None
    citation: str | None
    formatted_citation: str | None = None
    citation_style: str | None = None
    source_type: str | None
    source_url: str | None
    document_id: str | None
    page_number: int | None
    chunk_id: str | None
    claim_support: str | None
    applicability: str | None
    citation_verified: bool | None
    semantic_verified: bool | None
    core_claim_eligible: bool | None = None
    supported_claims: list[str] = Field(default_factory=list)
    scope_limits: list[str] = Field(default_factory=list)


class ProvenanceEntry(ContractModel):
    source_object: str | None
    source_id: str | None
    schema_version: str | None


class PaperAbstractSection(ContractModel):
    background: str
    objective: str
    methods: str
    expected_results: str
    observed_results: str | None = None
    conclusion: str | None = None
    limitations: str
    rendered_text: str
    # LLM 仅做语言润色的执行摘要；未配置/失败时保持 None，前端回退确定性 rendered_text。
    polished_text: str | None = None


class FinalPlanSelection(ContractModel):
    hypothesis_id: str
    round_index: int = 1
    revision_index: int = 0
    plan_id: str
    experiment_id: str | None = None
    selection_reason: str = "FIRST_ROUND_NO_ITERATION"
    iteration_occurred: bool = False
    fallback_applied: bool = False


class ScientificResearchPlan(ContractModel):
    """Post-experiment scientific deliverable derived from existing facts."""

    metadata: ScientificResearchPlanMetadata | None
    paper_title: str | None
    paper_abstract: str | PaperAbstractSection | None
    problem_statement: ProblemStatementSection | None
    rationale: RationaleSection | None
    technical_details: TechnicalDetailsSection | None
    dataset: DatasetSection | None
    methods: MethodsSection | None
    experiments: ExperimentsSection | None
    baselines: list[str] | None
    metrics: MetricsSection | None
    results: ResultsSection | None
    scientific_verdict: ScientificVerdictSnapshot | None
    experiment_validity: ExperimentValiditySnapshot | None
    references: list[ReferenceEntry] | None
    limitations: list[str] | None
    provenance: list[ProvenanceEntry] | None
    research_trace: list[ResearchTraceEntry] | None
    final_selection: FinalPlanSelection | None = None
