from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from boilermind.core.contracts.scientific_research_plan import (
    DatasetSection,
    ExperimentValiditySnapshot,
    ExperimentsSection,
    MethodsSection,
    MetricsSection,
    ModelResultSnapshot,
    ProblemStatementSection,
    ProvenanceEntry,
    RationaleSection,
    ReferenceEntry,
    ResearchTraceEntry,
    ResultsSection,
    ScientificResearchPlan,
    ScientificResearchPlanMetadata,
    ScientificVerdictSnapshot,
    TechnicalDetailsSection,
)
from boilermind.core.enums import ScientificVerdict


def _sections():
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    metadata = ScientificResearchPlanMetadata(
        schema_version="scientific.research.plan.v1",
        report_id="REPORT-001",
        generated_at=now,
        problem_id="PROBLEM-001",
        hypothesis_id="HYPOTHESIS-001",
        plan_id="PLAN-001",
        experiment_id="EXPERIMENT-001",
        report_status="complete",
    )
    problem = ProblemStatementSection(
        original_question="Does additional history improve prediction?",
        research_object="dynamic system prediction",
        target_variable="target_signal",
        objective="improve_prediction_quality",
        metrics=["metric_primary", "metric_secondary"],
        target_inference_reason="resolved from the frozen data schema",
        operating_condition="variable operating regime",
        manipulated_variables=("input_a",),
        observed_variables=("target_signal",),
        context_variables=("context_a",),
        research_goal="test the effect of historical context",
        success_criteria=["primary criterion is met"],
        constraints=["locked evaluation remains isolated"],
    )
    rationale = RationaleSection(
        research_significance="improve reliable prediction",
        hypothesis_statement="additional history improves prediction",
        mechanism_chain="more context -> stronger temporal representation",
        mechanism_steps=[],
        related_variables=["input_a"],
        applicability_conditions=["variable regime"],
        verification_intent="compare frozen history settings",
        expected_observation="primary metric improves",
        assumptions=[],
        counter_mechanisms=[],
        evidence_gaps=[],
        novelty_axis="history sensitivity",
        evidence_bundle_sha256="a" * 64,
        confirmation_criteria=("primary criterion is met",),
        falsification_criteria=("primary criterion is not met",),
    )
    technical = TechnicalDetailsSection(
        experiment_type="model_comparison",
        required_operations=["fit", "predict", "evaluate"],
        window_steps=40,
        prediction_horizon_steps=10,
        sampling_interval_seconds=1,
        random_seed=42,
        execution_requirements={},
        allowed_devices=["cpu"],
        reuse_checkpoint_models=[],
    )
    dataset = DatasetSection(
        source="verified_local_dataset",
        dataset_id="DATASET-001",
        dataset_hash="dataset-hash",
        dataset_path=None,
        target="target_signal",
        input_variables=["input_a"],
        train_split="chronological training split",
        validation_split="chronological validation split",
        locked_test_split="chronological locked split",
        scaler_fit_scope="training split only",
        chronological_split=True,
        sample_counts={"train": 100, "validation": 20, "test": 20},
    )
    methods = MethodsSection(
        objective="improve_prediction_quality",
        experimental_design="frozen comparison",
        baseline_description="predeclared reference method",
        intervention_description="increase historical context",
        control={"history_steps": 20},
        treatment={"history_steps": 40},
        recommended_models=["candidate_family_a"],
        executable_models=["candidate_family_b"],
        candidate_models=["candidate_family_b"],
        reference_models=("reference_method",),
        model_selection_rationale="matches the stated mechanism",
        model_substitution_reason="recommended dependency unavailable",
        primary_metric="metric_primary",
        secondary_metrics=("metric_secondary",),
        locked_test_used_for_selection=False,
        execution_backend="local_backend",
        allow_partial_failure=False,
        max_runtime_per_model=60.0,
        max_epochs=None,
        confirmation_criteria=["primary criterion is met"],
        falsification_criteria=["primary criterion is not met"],
    )
    model_result = ModelResultSnapshot(
        model_name="candidate_family_b",
        fit_success=True,
        fit_converged=True,
        runtime_seconds=1.25,
        validation_metrics={"metric_primary": 0.2},
        locked_test_metrics={"metric_primary": 0.25},
        warnings=(),
        failure_reason=None,
        sample_counts={"train": 100, "validation": 20, "test": 20},
        random_seed=42,
        device="cpu",
        artifact_provenance={},
    )
    experiments = ExperimentsSection(
        experiment_id="EXPERIMENT-001",
        status="completed",
        started_at=now,
        completed_at=now,
        model_results=[model_result],
        execution_notes=[],
        artifacts=[],
    )
    metrics = MetricsSection(
        planned_metrics=["metric_primary", "metric_secondary"],
        primary_metric="metric_primary",
        secondary_metrics=["metric_secondary"],
        validation_metrics_by_model={
            "candidate_family_b": {"metric_primary": 0.2}
        },
        locked_test_metrics_by_model={
            "candidate_family_b": {"metric_primary": 0.25}
        },
        baseline_metrics={"metric_primary": 0.3},
        control_metrics={},
        treatment_metrics={},
        metric_deltas={},
    )
    results = ResultsSection(
        overall_metrics={"metric_primary": 0.25},
        baseline_metrics={"metric_primary": 0.3},
        candidate_locked_test_metrics={
            "candidate_family_b": {"metric_primary": 0.25}
        },
        control_metrics={},
        treatment_metrics={},
        metric_deltas={},
    )
    verdict = ScientificVerdictSnapshot(
        verdict=ScientificVerdict.SUPPORTED,
        rationale="the predeclared criterion is supported",
        achieved_criteria=("primary criterion is met",),
        failed_criteria=(),
        source_hypothesis_id="HYPOTHESIS-001",
        source_experiment_id="EXPERIMENT-001",
    )
    validity = ExperimentValiditySnapshot(
        experiment_valid=True,
        execution_valid=True,
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
        issues=(),
        validity_source="ExperimentResult+ExperimentAudit",
    )
    trace = ResearchTraceEntry(
        plan_id="PLAN-001",
        experiment_id="EXPERIMENT-001",
        status="completed",
        metrics={"metric_primary": 0.25},
        target_met=True,
        reason="predeclared criterion is met",
    )
    return {
        "metadata": metadata,
        "problem": problem,
        "rationale": rationale,
        "technical": technical,
        "dataset": dataset,
        "methods": methods,
        "experiments": experiments,
        "metrics": metrics,
        "results": results,
        "verdict": verdict,
        "validity": validity,
        "trace": trace,
    }


def _plan() -> ScientificResearchPlan:
    sections = _sections()
    return ScientificResearchPlan(
        metadata=sections["metadata"],
        paper_title="Historical context and prediction quality",
        paper_abstract="A deterministic summary of recorded research facts.",
        problem_statement=sections["problem"],
        rationale=sections["rationale"],
        technical_details=sections["technical"],
        dataset=sections["dataset"],
        methods=sections["methods"],
        experiments=sections["experiments"],
        baselines=["reference_method"],
        metrics=sections["metrics"],
        results=sections["results"],
        scientific_verdict=sections["verdict"],
        experiment_validity=sections["validity"],
        references=[
            ReferenceEntry(
                evidence_id="EVIDENCE-001",
                title="Verified source",
                citation="Verified citation",
                source_type="verified_repository",
                source_url=None,
                document_id="DOCUMENT-001",
                page_number=1,
                chunk_id="CHUNK-001",
                claim_support="direct",
                applicability="direct",
                citation_verified=True,
                semantic_verified=True,
            )
        ],
        limitations=["single dataset"],
        provenance=[
            ProvenanceEntry(
                source_object="ScientificResult",
                source_id="EXPERIMENT-001",
                schema_version="current",
            )
        ],
        research_trace=[sections["trace"]],
    )


def test_all_sections_instantiate():
    sections = _sections()
    assert all(value is not None for value in sections.values())
    assert _plan().metadata is not None


def test_research_trace_entry_serializes_to_json():
    trace = _sections()["trace"]
    assert '"experiment_id":"EXPERIMENT-001"' in trace.model_dump_json()


def test_metrics_section_keeps_planned_validation_and_locked_test_separate():
    metrics = _sections()["metrics"]
    assert metrics.planned_metrics == ["metric_primary", "metric_secondary"]
    assert metrics.validation_metrics_by_model == {
        "candidate_family_b": {"metric_primary": 0.2}
    }
    assert metrics.locked_test_metrics_by_model == {
        "candidate_family_b": {"metric_primary": 0.25}
    }


def test_scientific_verdict_snapshot_is_frozen():
    verdict = _sections()["verdict"]
    with pytest.raises(ValidationError):
        verdict.verdict = ScientificVerdict.FALSIFIED


def test_experiment_validity_snapshot_is_frozen():
    validity = _sections()["validity"]
    with pytest.raises(ValidationError):
        validity.experiment_valid = False


def test_model_dump_and_json_serialization():
    plan = _plan()
    payload = plan.model_dump(mode="json")
    encoded = plan.model_dump_json()
    assert payload["scientific_verdict"]["verdict"] == "supported"
    assert payload["metrics"]["primary_metric"] == "metric_primary"
    assert '"research_trace"' in encoded


def test_missing_required_top_level_fields_fail_validation():
    with pytest.raises(ValidationError):
        ScientificResearchPlan.model_validate({"paper_title": None})
