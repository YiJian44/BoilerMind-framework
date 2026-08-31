from __future__ import annotations

from datetime import datetime, timezone

import pytest

from boilermind.core.contracts import (
    ApprovalStatus,
    ComparisonLevel,
    CurrentObservationBundle,
    EvidenceTier,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
    NextRoundProposalBundle,
    ResearchProblemSpec,
)
from boilermind.experiment_memory import (
    ExperimentMemoryStore,
    build_opportunity_map,
    derive_experiment_observations,
    import_experiment_history,
    retrieve_experiment_memory,
    match_post_experiment_literature,
    extract_current_observations,
    enforce_generation_quota,
)
from boilermind.experiment_memory.persistence import (
    build_empirical_capability_profile,
    build_next_round_proposal,
    persist_experiment_outcome,
)
from boilermind.core.contracts import ExperimentAudit, ExperimentResult, ScientificResult
from boilermind.core.contracts import ExperimentContract
from boilermind.core.enums import ExperimentStatus, ScientificVerdict


def _record(experiment_id: str, *, mode="direct_volume", horizon=80, tier=EvidenceTier.AUDITED_CONFIRMATORY):
    return HistoricalExperimentRecord(
        experiment_id=experiment_id,
        series_id="SERIES-1",
        hypothesis_id="H-1",
        source_type="test",
        source_path="fixture.json",
        source_sha256="a" * 64,
        source_locator="fixture:1",
        scope=ExperimentScopeSignature(
            target_variable="steam_volumetric_flow",
            target_unit="m3/s",
            prediction_mode=mode,
            thermodynamic_standard="IF97",
            dataset_sha256="b" * 64,
            window_steps=20,
            prediction_horizon_steps=horizon,
            split_policy="chronological",
        ),
        candidate_models=["ridge"],
        verdict="SUPPORTED",
        evidence_tier=tier,
        raw_result="整体支持；ramp_down 条件下结果存在边界。",
        raw_limitations="独立时间段尚未复验。",
    )


def _problem():
    return ResearchProblemSpec(
        problem_id="P-1",
        original_question="IF97 direct-V 20分钟预测",
        research_object="锅炉",
        target_variable="steam_volumetric_flow",
        operating_condition="深度调峰",
        research_goal="预测未来20分钟体积流量",
    )


def test_observations_split_supported_and_boundary():
    observations = derive_experiment_observations(_record("EXP-1"))
    assert len(observations) >= 2
    assert {item.observation_type.value for item in observations} >= {"SUPPORTED", "BOUNDARY_CONDITION"}


def test_insufficient_evidence_never_becomes_falsified():
    record = _record("EXP-INSUFFICIENT")
    record.verdict = "INSUFFICIENT_EVIDENCE"
    record.raw_result = (
        "verdict=INSUFFICIENT_EVIDENCE; feedback=Experiment cannot "
        "provide valid ranking feedback because evidence is insufficient "
        "or audit failed."
    )
    observations = derive_experiment_observations(record)
    assert observations
    assert all(
        item.observation_type.value != "FALSIFIED"
        for item in observations
    )


def test_runtime_result_is_one_structured_observation():
    record = _record("EXP-RUNTIME-STRUCTURED")
    record.source_type = "boilermind_runtime"
    record.audit_status = "PASSED"
    record.verdict = "FALSIFIED"
    record.metrics = {
        "candidate_locked_test_metrics": {
            "ridge": {"MAE": 1.0},
            "persistence": {"MAE": 2.0},
        }
    }
    record.raw_result = (
        "verdict=FALSIFIED; metrics={'MAE': 1.0}; better=ridge"
    )

    observations = derive_experiment_observations(record)

    assert len(observations) == 1
    assert observations[0].observation_type.value == "FALSIFIED"
    assert observations[0].supporting_metrics == record.metrics
    assert "better=" not in observations[0].claim
    assert not observations[0].invalid_for_scientific_synthesis


def test_failed_runtime_result_is_engineering_only():
    record = _record("EXP-RUNTIME-FAILED")
    record.source_type = "boilermind_runtime"
    record.audit_status = "FAILED"
    record.evidence_tier = EvidenceTier.ENGINEERING_FAILURE

    observation = derive_experiment_observations(record)[0]

    assert observation.observation_type.value == "ENGINEERING_FAILURE"
    assert observation.invalid_for_scientific_synthesis
    assert observation.reuse_policy == "ENGINEERING_ONLY"


def test_structured_scope_filter_blocks_mass_to_direct_volume(tmp_path):
    direct = _record("EXP-DIRECT")
    mass = _record("EXP-MASS", mode="mass")
    observations = [*derive_experiment_observations(direct), *derive_experiment_observations(mass)]
    store = ExperimentMemoryStore(tmp_path)
    store.replace_all([direct, mass], observations, [])
    bundle = retrieve_experiment_memory(
        _problem(),
        {
            "dataset_contract": {"dataset_hash": "b" * 64},
            "prediction_horizon_steps": 80,
            "enabled_experiment_models": ["ridge"],
            "reference_model": "persistence",
        },
        store,
    )
    assert "EXP-DIRECT" in bundle.completed_experiment_ids
    assert "EXP-MASS" not in bundle.completed_experiment_ids


def test_planned_experiment_never_becomes_scientific_support(tmp_path):
    planned = _record("EXP-PLANNED", tier=EvidenceTier.PLANNED_NOT_EXECUTED)
    planned.verdict = "NOT_EXECUTED"
    store = ExperimentMemoryStore(tmp_path)
    store.replace_all([planned], [], [])
    bundle = retrieve_experiment_memory(_problem(), {"enabled_experiment_models": ["ridge"]}, store)
    assert not bundle.completed_experiment_ids


def test_opportunity_map_uses_experiment_memory_not_literature(tmp_path):
    record = _record("EXP-1")
    observations = derive_experiment_observations(record)
    store = ExperimentMemoryStore(tmp_path)
    store.replace_all([record], observations, [])
    bundle = retrieve_experiment_memory(_problem(), {"enabled_experiment_models": ["ridge"]}, store)
    result = build_opportunity_map(bundle, CurrentObservationBundle(problem_id="P-1"), {"enabled_experiment_models": ["ridge"]})
    assert result.opportunities
    assert all("LITERATURE_INSPIRATION" not in [trigger.value for trigger in item.trigger_types] for item in result.opportunities)


def test_next_round_defaults_to_human_approval():
    bundle = NextRoundProposalBundle(proposal_id="NRP-1", source_experiment_id="EXP-1")
    assert bundle.approval_status == ApprovalStatus.PENDING_HUMAN_APPROVAL
    with pytest.raises(ValueError, match="approved_next_round_requires_candidate"):
        NextRoundProposalBundle(proposal_id="NRP-2", source_experiment_id="EXP-1", approval_status=ApprovalStatus.APPROVED)


def test_store_is_immutable_for_existing_experiment_id(tmp_path):
    store = ExperimentMemoryStore(tmp_path)
    first = _record("EXP-1")
    store.replace_all([first], derive_experiment_observations(first), [])
    changed = _record("EXP-1")
    changed.raw_result = "被篡改"
    with pytest.raises(ValueError, match="immutable_experiment_conflict"):
        store.append_record(changed, [])


def test_post_experiment_literature_match_is_not_formal_citation():
    observation = derive_experiment_observations(_record("EXP-1"))[0]
    relations = match_post_experiment_literature([observation], {
        "evidence": [{
            "evidence_id": "E-1",
            "document_id": "DOC-1",
            "title": "整体支持的实验方法",
            "text": observation.claim,
            "claim_support": "direct",
            "applicability": "medium",
            "page_number": 2,
            "chunk_id": "C-1",
            "metadata_status": "needs_review",
        }]
    })
    assert relations
    assert relations[0].relationship == "METHOD_RELATED"
    assert relations[0].formatted_citation is None
    assert relations[0].metadata_verified is False


def test_runtime_outcome_persists_and_requires_next_round_approval(tmp_path):
    now = datetime.now(timezone.utc)
    result = ExperimentResult(
        experiment_id="EXP-RUNTIME",
        problem_id="P-1",
        hypothesis_id="H-1",
        status=ExperimentStatus.COMPLETED,
        metrics={"MAE": 1.0},
        started_at=now,
        completed_at=now,
    )
    audit = ExperimentAudit(
        experiment_id="EXP-RUNTIME",
        execution_valid=True,
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
    )
    scientific = ScientificResult(
        hypothesis_id="H-1",
        experiment_id="EXP-RUNTIME",
        verdict=ScientificVerdict.SUPPORTED,
        rationale="预声明标准已满足",
    )
    store = ExperimentMemoryStore(tmp_path)
    record, observations = persist_experiment_outcome({
        "experiment_result": result,
        "audit": audit,
        "scientific_result": scientific,
    }, store)
    proposal = build_next_round_proposal(record, observations)
    assert store.load_records()[0].experiment_id == "EXP-RUNTIME"
    assert proposal.approval_status == ApprovalStatus.PENDING_HUMAN_APPROVAL


def test_runtime_outcome_persists_complete_contract_scope(tmp_path):
    now = datetime.now(timezone.utc)
    contract = ExperimentContract(
        experiment_id="EXP-SCOPE",
        problem_id="P-1",
        hypothesis_id="H-1",
        plan_id="PLAN-H-1",
        dataset_id="DATA-1",
        dataset_hash="c" * 64,
        input_variables=["load"],
        target_variable="steam_volumetric_flow",
        train_split="chronological_train",
        validation_split="chronological_validation",
        test_split="locked_test",
        baseline_models=["persistence"],
        candidate_models=["ridge"],
        metrics=["MAE"],
        confirmation_criteria=["candidate_better"],
        falsification_criteria=["candidate_not_better"],
        sampling_interval_seconds=15,
        window_steps=20,
        prediction_horizon_steps=80,
    )
    result = ExperimentResult(
        experiment_id="EXP-SCOPE",
        problem_id="P-1",
        hypothesis_id="H-1",
        plan_id="PLAN-H-1",
        status=ExperimentStatus.COMPLETED,
        metrics={"MAE": 1.0},
        started_at=now,
        completed_at=now,
    )
    audit = ExperimentAudit(
        experiment_id="EXP-SCOPE",
        execution_valid=True,
        dataset_frozen=True,
        leakage_check_passed=True,
        baseline_valid=True,
        metric_check_passed=True,
    )
    scientific = ScientificResult(
        hypothesis_id="H-1",
        experiment_id="EXP-SCOPE",
        verdict=ScientificVerdict.SUPPORTED,
        rationale="通过",
    )
    record, _ = persist_experiment_outcome(
        {
            "experiment_contract": contract,
            "experiment_result": result,
            "audit": audit,
            "scientific_result": scientific,
        },
        ExperimentMemoryStore(tmp_path),
    )
    assert record.scope.target_variable == "steam_volumetric_flow"
    assert record.scope.prediction_horizon_steps == 80
    assert record.scope.window_steps == 20
    assert record.scope.dataset_sha256 == "c" * 64
    assert record.scope.split_policy == (
        "chronological_train|chronological_validation|locked_test"
    )


def test_same_dataset_with_unknown_hard_scope_is_not_directly_comparable(tmp_path):
    incomplete = _record("EXP-INCOMPLETE")
    incomplete.scope = ExperimentScopeSignature(dataset_sha256="b" * 64)
    store = ExperimentMemoryStore(tmp_path)
    store.replace_all(
        [incomplete], derive_experiment_observations(incomplete), []
    )
    bundle = retrieve_experiment_memory(
        _problem(),
        {
            "dataset_contract": {"dataset_hash": "b" * 64},
            "prediction_horizon_steps": 80,
        },
        store,
    )
    assert not bundle.directly_comparable
    assert bundle.conditionally_comparable


def test_current_observations_are_runtime_facts_not_scientific_conclusions():
    bundle = extract_current_observations(_problem(), {
        "dataset_contract": {"dataset_id": "D-1", "dataset_hash": "a" * 64, "row_count": 100},
        "enabled_experiment_models": ["ridge"],
        "supported_experiment_operations": ["model_comparison"],
    })
    assert bundle.observations
    assert all(item["scientific_conclusion"] is False for item in bundle.observations)


def test_literature_candidates_are_capped_at_ten_percent_slot():
    candidates = [{"hypothesis_id": f"H{i}", "trigger_types": ["LITERATURE_INSPIRATION"]} for i in range(3)]
    accepted, rejected = enforce_generation_quota(candidates)
    assert len(accepted) == 1
    assert len(rejected) == 2


def test_empirical_capability_excludes_mentions_plans_and_unverified_claims():
    exploratory = _record("EXP-RUN", tier=EvidenceTier.AUDITED_EXPLORATORY)
    exploratory.audit_status = "PASSED_EXPLORATORY_ONLY"
    exploratory.metrics = {"runtime_seconds": 1.25}

    legacy = _record("EXP-LEGACY", tier=EvidenceTier.LEGACY_INFORMATIVE)
    planned = _record("EXP-PLAN", tier=EvidenceTier.PLANNED_NOT_EXECUTED)
    unverified = _record("EXP-UNVERIFIED", tier=EvidenceTier.AUDITED_CONFIRMATORY)
    unverified.audit_status = "LEGACY_LOG_REVIEWED"

    profile = build_empirical_capability_profile(
        [exploratory, legacy, planned, unverified]
    )

    assert len(profile.models) == 1
    assert profile.models[0].model_id == "ridge"
    assert profile.models[0].run_count == 1
    assert profile.models[0].success_count == 1
    assert profile.models[0].runtime_seconds == [1.25]
