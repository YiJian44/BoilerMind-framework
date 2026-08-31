from datetime import datetime, timezone

from boilermind.core.contracts.evidence import EvidenceBundle, VerifiedEvidence
from boilermind.core.contracts.experiment import (
    ExperimentAudit,
    ExperimentContract,
    ExperimentPlan,
    ExperimentResult,
    ModelExperimentRecord,
    ScientificResult,
)
from boilermind.core.contracts.hypothesis import MechanismStep, ScientificHypothesis
from boilermind.core.contracts.research_problem import ResearchProblemSpec
from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
    ExperimentStatus,
    MechanismSupportType,
    ScientificVerdict,
)
from boilermind.reporting import (
    ScientificResearchPlanGenerator,
    ScientificResearchPlanGeneratorInput,
    ScientificResearchPlanService,
)
from boilermind.reporting.scientific_research_plan_renderer import ScientificResearchPlanRenderer
from boilermind.reporting.scientific_research_plan_renderer import _localize_report_text


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def _input(*, valid=True, verdict=ScientificVerdict.SUPPORTED):
    problem = ResearchProblemSpec(
        problem_id="PROBLEM-001",
        original_question="Does historical context improve prediction?",
        research_object="dynamic system prediction",
        target_variable="target_signal",
        objective="improve_prediction_quality",
        metrics=["metric_primary", "metric_secondary"],
        target_inference_reason="resolved from the frozen schema",
        operating_condition="variable regime",
        manipulated_variables=["input_a"],
        observed_variables=["target_signal"],
        context_variables=["context_a"],
        research_goal="test historical context",
        success_criteria=["primary criterion met"],
        constraints=["locked evaluation isolated"],
    )
    evidence = VerifiedEvidence(
        evidence_id="EVIDENCE-001",
        problem_id="PROBLEM-001",
        source_type="verified_repository",
        title="Existing verified reference",
        source_url="https://example.test/reference",
        citation="Existing citation text",
        formatted_citation="AUTHOR A. Existing verified reference[J]. Journal, 2024, 1(1): 1-2.",
        text="Existing verified evidence text.",
        retrieval_score=0.9,
        document_id="DOCUMENT-001",
        chunk_id="CHUNK-001",
        page_number=1,
        citation_verified=True,
        semantic_verified=True,
        claim_support=ClaimSupport.DIRECT,
        applicability=ApplicabilityLevel.HIGH,
        core_claim_eligible=True,
        verification_rationale="verified by existing pipeline",
    )
    bundle = EvidenceBundle(
        bundle_id="BUNDLE-001",
        problem_id="PROBLEM-001",
        evidence=[evidence],
        created_at=NOW,
        sha256="a" * 64,
    )
    hypothesis = ScientificHypothesis(
        hypothesis_id="HYPOTHESIS-001",
        problem_id="PROBLEM-001",
        title="Historical context and prediction quality",
        research_significance="improve reliable prediction",
        hypothesis="additional history improves prediction",
        mechanism_chain="more context -> stronger temporal representation",
        mechanism_steps=[
            MechanismStep(
                step=1,
                statement="additional history provides temporal context",
                support_type=MechanismSupportType.VERIFIED_EVIDENCE,
                evidence_ids=["EVIDENCE-001"],
            )
        ],
        related_variables=["input_a"],
        applicability_conditions=["variable regime"],
        verification_intent="compare frozen history settings",
        expected_observation="primary metric improves",
        confirmation_criteria=["primary criterion met"],
        falsification_criteria=["primary criterion not met"],
        evidence_gaps=["single dataset"],
        assumptions=[],
        counter_mechanisms=[],
        novelty_axis="history sensitivity",
        evidence_bundle_sha256="a" * 64,
    )
    plan = ExperimentPlan(
        plan_id="PLAN-001",
        hypothesis_id="HYPOTHESIS-001",
        problem_id="PROBLEM-001",
        candidate_models=["candidate_method"],
        recommended_models=["candidate_method"],
        executable_models=["candidate_method"],
        reference_models=["reference_method"],
        model_substitution_reason="",
        control={"history_steps": 10},
        treatment={"history_steps": 20},
        target="target_signal",
        prediction_horizon_steps=5,
        primary_metric="metric_primary",
        secondary_metrics=["metric_secondary"],
        required_operations=["fit", "predict", "evaluate"],
        model_selection_rationale="matches the stated mechanism",
        dataset_path="C:/sensitive/local/path/data.csv",
        window_steps=20,
        sampling_interval_seconds=1,
        model_candidates=["candidate_method"],
        reference_model="reference_method",
        selection_metric="metric_primary",
        locked_test_used_for_selection=False,
        random_seed=42,
        objective="improve_prediction_quality",
        experimental_design="frozen comparison",
        baseline_description="predeclared reference method",
        intervention_description="increase historical context",
        required_variables=["input_a"],
        metrics=["metric_primary", "metric_secondary"],
        expected_observation="primary metric improves",
        confirmation_criteria=["primary criterion met"],
        falsification_criteria=["primary criterion not met"],
    )
    contract = ExperimentContract(
        experiment_id="EXPERIMENT-001",
        problem_id="PROBLEM-001",
        hypothesis_id="HYPOTHESIS-001",
        plan_id="PLAN-001",
        experiment_type="model_comparison",
        control={"history_steps": 10},
        treatment={"history_steps": 20},
        primary_metric="metric_primary",
        secondary_metrics=["metric_secondary"],
        reference_models=["reference_method"],
        model_selection_rationale="matches the stated mechanism",
        recommended_models=["candidate_method"],
        executable_models=["candidate_method"],
        model_substitution_reason="",
        prediction_horizon_steps=5,
        sampling_interval_seconds=1,
        window_steps=20,
        locked_test_used_for_selection=False,
        required_operations=["fit", "predict", "evaluate"],
        execution_requirements={},
        dataset_id="DATASET-001",
        dataset_hash="dataset-hash",
        input_variables=["input_a"],
        target_variable="target_signal",
        train_split="chronological train",
        validation_split="chronological validation",
        test_split="chronological locked test",
        baseline_models=["reference_method"],
        candidate_models=["candidate_method"],
        metrics=["metric_primary", "metric_secondary"],
        confirmation_criteria=["primary criterion met"],
        falsification_criteria=["primary criterion not met"],
        random_seed=42,
    )
    record = ModelExperimentRecord(
        model_name="candidate_method",
        fit_success=True,
        fit_converged=True,
        runtime_seconds=1.5,
        validation_metrics={"metric_primary": 0.2},
        locked_test_metrics={"metric_primary": 0.25},
        train_samples=100,
        validation_samples=20,
        test_samples=20,
        random_seed=42,
        artifact_provenance={"source": "existing runner"},
    )
    result = ExperimentResult(
        experiment_id="EXPERIMENT-001",
        problem_id="PROBLEM-001",
        hypothesis_id="HYPOTHESIS-001",
        plan_id="PLAN-001",
        status=ExperimentStatus.COMPLETED,
        metrics={"metric_primary": 0.25},
        baseline_metrics={"metric_primary": 0.3},
        candidate_locked_test_metrics={
            "candidate_method": {"metric_primary": 0.25}
        },
        model_records={"candidate_method": record},
        experiment_valid=valid,
        experiment_validity_issues=[] if valid else ["audit did not pass"],
        started_at=NOW,
        completed_at=NOW,
    )
    audit = ExperimentAudit(
        experiment_id="EXPERIMENT-001",
        execution_valid=valid,
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
        issues=[] if valid else ["execution invalid"],
    )
    scientific = ScientificResult(
        hypothesis_id="HYPOTHESIS-001",
        experiment_id="EXPERIMENT-001",
        verdict=verdict,
        rationale="existing scientific determination",
        achieved_criteria=[] if not valid else ["primary criterion met"],
        failed_criteria=[] if valid else ["valid experiment required"],
    )
    return ScientificResearchPlanGeneratorInput(
        research_problem=problem,
        evidence_bundle=bundle,
        hypothesis=hypothesis,
        experiment_plan=plan,
        experiment_contract=contract,
        experiment_result=result,
        experiment_audit=audit,
        scientific_result=scientific,
        research_trace=[
            {
                "plan_id": "PLAN-001",
                "experiment_id": "EXPERIMENT-001",
                "status": "completed",
                "metrics": {"metric_primary": 0.25},
                "target_met": valid,
                "reason": "existing trace reason",
            }
        ],
    )


def test_complete_input_generates_scientific_research_plan():
    report = ScientificResearchPlanGenerator().generate(_input())
    assert report.metadata.report_status == "complete"
    assert report.experiments.model_results[0].model_name == "candidate_method"
    assert report.dataset.dataset_path is None
    assert report.paper_abstract
    assert report.rationale.competing_hypotheses
    for hypothesis in report.rationale.competing_hypotheses:
        experiment_plan = hypothesis["experiment_plan"]
        assert experiment_plan["objective"]
        assert experiment_plan["confirmation_rule"]
        assert experiment_plan["falsification_rule"]
        assert len(experiment_plan["execution_steps"]) == 5
        assert experiment_plan["planned_outputs"]
    assert report.results.protocol_selected_model == "candidate_method"
    assert report.results.locked_test_best_model == "candidate_method"


def test_reader_report_hides_internal_audit_details_and_reference_hypotheses():
    report = ScientificResearchPlanGenerator().generate(_input())

    assert [item["model"] for item in report.rationale.competing_hypotheses] == [
        "candidate_method"
    ]
    markdown = ScientificResearchPlanRenderer().render_markdown(report)
    assert "审计附录" not in markdown
    assert "运行 ID：" not in markdown
    assert "数据 SHA-256" not in markdown


def test_report_renderer_localizes_execution_terms_and_verdicts():
    rendered = _localize_report_text(
        "UNIFIED_EXPERIMENT_EXECUTION; target_profile:steam_volumetric_flow; "
        "locked_test_not_used_for_selection; locked_test; falsified; "
        "any_candidate_better_than_reference_on:MAE; gru"
    )

    assert rendered == (
        "统一实验执行; 目标变量：蒸汽体积流量 V; 锁定测试集不参与模型选择; 锁定测试集; "
        "未获支持; 任一候选模型在 MAE 上优于参考模型; GRU"
    )


def test_word_renderer_strips_markdown_markers_and_preserves_numbering(tmp_path):
    from docx import Document

    report = ScientificResearchPlanGenerator().generate(_input())
    output = tmp_path / "plan.docx"
    ScientificResearchPlanRenderer().render_docx(report, output)
    text = "\n".join(paragraph.text for paragraph in Document(output).paragraphs)

    assert "**" not in text
    assert "1. 冻结数据：校验数据集标识与SHA256" in text
    assert "1. 对协议选择模型开展多随机种子复验" in text


def test_id_mismatch_returns_failed_report_with_warning():
    data = _input()
    data.experiment_plan = data.experiment_plan.model_copy(
        update={"problem_id": "DIFFERENT-PROBLEM"}
    )
    report = ScientificResearchPlanGenerator().generate(data)
    assert report.metadata.report_status == "failed"
    assert report.paper_abstract is None
    assert "consistency_warning:problem_id:mismatch" in report.limitations


def test_scientific_verdict_is_copied_without_recalculation():
    data = _input(verdict=ScientificVerdict.PARTIALLY_SUPPORTED)
    report = ScientificResearchPlanGenerator().generate(data)
    assert report.scientific_verdict.verdict is data.scientific_result.verdict


def test_invalid_experiment_does_not_become_falsified():
    report = ScientificResearchPlanGenerator().generate(
        _input(valid=False, verdict=ScientificVerdict.INSUFFICIENT_EVIDENCE)
    )
    assert report.experiment_validity.experiment_valid is False
    assert report.scientific_verdict.verdict is ScientificVerdict.INSUFFICIENT_EVIDENCE
    assert report.scientific_verdict.verdict is not ScientificVerdict.FALSIFIED


def test_references_only_copy_existing_evidence():
    data = _input()
    report = ScientificResearchPlanGenerator().generate(data)
    assert len(report.references) == len(data.evidence_bundle.evidence) == 1
    assert report.references[0].title == "Existing verified reference"
    assert report.references[0].citation == "Existing citation text"
    assert report.references[0].formatted_citation.startswith("AUTHOR A.")
    assert report.references[0].citation_style == "GB/T 7714—2015"
    assert report.references[0].source_url == "https://example.test/reference"


def test_metrics_come_from_contract_and_result():
    data = _input()
    report = ScientificResearchPlanGenerator().generate(data)
    assert report.metrics.planned_metrics == data.experiment_contract.metrics
    assert report.metrics.validation_metrics_by_model == {
        "candidate_method": {"metric_primary": 0.2}
    }
    assert report.metrics.locked_test_metrics_by_model == {
        "candidate_method": {"metric_primary": 0.25}
    }
    assert report.results.overall_metrics == data.experiment_result.metrics


def test_standardized_sections_are_grounded_in_existing_contracts():
    report = ScientificResearchPlanGenerator().generate(_input())
    assert report.problem_statement.current_limitation
    assert report.rationale.innovation_point
    assert report.rationale.reasoning_chain
    assert report.technical_details.technical_stack
    assert report.dataset.source_compliance_status == "VERIFIED"
    assert report.dataset.target == "target_signal"
    assert report.paper_abstract.expected_results
    assert report.methods.implementation_steps
    assert report.experiments.baselines
    assert report.experiments.metric_definitions
    assert report.results.feasibility_basis == "ACTUAL_EXECUTION"
    assert report.references[0].citation_verified is True


def test_service_writes_json_markdown_word_and_manifest(tmp_path):
    response = ScientificResearchPlanService().generate(
        run_id="RUN-REPORT-TEST",
        output_dir=tmp_path,
        input_data=_input().model_copy(update={"run_id": "RUN-REPORT-TEST"}),
    )
    assert response.status == "GENERATED"
    for path in (
        response.json_path, response.markdown_path,
        response.word_path, response.manifest_path,
    ):
        assert path is not None
        assert __import__("pathlib").Path(path).is_file()
    assert __import__("pathlib").Path(response.word_path).read_bytes()[:2] == b"PK"
    markdown = __import__("pathlib").Path(response.markdown_path).read_text(encoding="utf-8")
    assert "《科研假设与研究计划》" in markdown
    assert "参考文献" in markdown
    assert "GB/T 7714—2015" in markdown
    assert "[1] AUTHOR A." in markdown
    assert "验证集（用于选择）" in markdown
    assert "锁定测试集（仅用于泛化评价）" in markdown
    assert "协议选择模型" in markdown
    assert "Background:" not in markdown
