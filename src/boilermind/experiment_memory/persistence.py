from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boilermind.core.contracts import (
    EmpiricalCapabilityProfile,
    EvidenceTier,
    ExperimentObservation,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
    ModelCapabilityPerformance,
    NextRoundCandidate,
    NextRoundProposalBundle,
    ObservationType,
)

from .observations import derive_experiment_observations
from .store import ExperimentMemoryStore


def _hash_payload(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _value(payload: Any, name: str, default=None):
    return getattr(payload, name, payload.get(name, default) if isinstance(payload, dict) else default)


def record_from_experiment_outcome(outcome: dict[str, Any], source_path: str = "runtime_generated") -> HistoricalExperimentRecord:
    contract = outcome.get("experiment_contract")
    result = outcome["experiment_result"]
    audit = outcome["audit"]
    scientific = outcome["scientific_result"]
    observable = outcome.get("observable_premise_result")
    feedback = outcome.get("feedback")
    experiment_id = str(_value(result, "experiment_id"))
    audit_valid = bool(_value(audit, "execution_valid", False))
    model_records = _value(result, "model_records", {}) or {}
    dataset_hash = None
    candidate_models: list[str] = []
    known_issues = list(_value(audit, "issues", []) or [])
    for model_id, model_record in model_records.items():
        candidate_models.append(str(model_id))
        dataset_hash = dataset_hash or _value(model_record, "dataset_sha256")
        known_issues.extend(list(_value(model_record, "warnings", []) or []))
        if _value(model_record, "failure_reason"):
            known_issues.append(str(_value(model_record, "failure_reason")))
    verdict = str(getattr(_value(scientific, "verdict"), "value", _value(scientific, "verdict", "INSUFFICIENT_EVIDENCE"))).upper()
    tier = EvidenceTier.AUDITED_CONFIRMATORY if audit_valid else EvidenceTier.ENGINEERING_FAILURE
    metrics = dict(_value(result, "metrics", {}) or {})
    metrics["candidate_locked_test_metrics"] = dict(
        _value(result, "candidate_locked_test_metrics", {}) or {}
    )
    metrics["regime_metrics"] = dict(
        _value(result, "regime_metrics", {}) or {}
    )
    conclusion_scope = str(
        _value(result, "conclusion_scope", "full_hypothesis")
    )
    if observable is not None:
        premise_verdict = str(getattr(
            _value(observable, "verdict"), "value",
            _value(observable, "verdict", "INSUFFICIENT_EVIDENCE"),
        )).upper()
        verdict = f"OBSERVABLE_PREMISE_{premise_verdict}"
    raw_result = f"verdict={verdict}; metrics={metrics}; rationale={_value(scientific, 'rationale', '')}"
    if feedback is not None:
        raw_result += f"; feedback={_value(feedback, 'rationale', '')}"
    return HistoricalExperimentRecord(
        experiment_id=experiment_id,
        series_id=str(_value(result, "hypothesis_id")),
        hypothesis_id=str(_value(result, "hypothesis_id")),
        source_type="boilermind_runtime",
        source_path=source_path,
        source_sha256=_hash_payload(raw_result),
        source_locator=f"experiment_id:{experiment_id}",
        scope=ExperimentScopeSignature(
            target_variable=_value(contract, "target_variable"),
            target_definition=_value(contract, "target_variable"),
            dataset_id=_value(contract, "dataset_id"),
            dataset_sha256=(
                _value(contract, "dataset_hash") or dataset_hash
            ),
            prediction_mode=(
                "direct_forecast"
                if _value(contract, "prediction_horizon_steps")
                else None
            ),
            sampling_interval_seconds=_value(
                contract, "sampling_interval_seconds"
            ),
            window_steps=_value(contract, "window_steps"),
            prediction_horizon_steps=_value(
                contract, "prediction_horizon_steps"
            ),
            split_policy="|".join(
                str(value)
                for value in (
                    _value(contract, "train_split"),
                    _value(contract, "validation_split"),
                    _value(contract, "test_split"),
                )
                if value
            ) or None,
        ),
        candidate_models=sorted(candidate_models),
        locked_test_used_for_selection=False,
        metrics=metrics,
        verdict=verdict,
        verdict_scope=[conclusion_scope],
        evidence_tier=tier,
        audit_status="PASSED" if audit_valid else "FAILED",
        known_issues=sorted(set(known_issues)),
        reproducibility_status="RUNTIME_CAPTURED",
        raw_result=raw_result,
        raw_limitations="; ".join(sorted(set(known_issues + (
            ["MECHANISM_NOT_TESTED"]
            if conclusion_scope == "problem_observable_premise_only" else []
        )))),
    )


def persist_experiment_outcome(outcome: dict[str, Any], store: ExperimentMemoryStore, source_path: str = "runtime_generated") -> tuple[HistoricalExperimentRecord, list[ExperimentObservation]]:
    record = record_from_experiment_outcome(outcome, source_path)
    observations = derive_experiment_observations(record)
    store.append_record(record, observations)
    return record, observations


def build_next_round_proposal(record: HistoricalExperimentRecord, observations: list[ExperimentObservation]) -> NextRoundProposalBundle:
    scientific = [item for item in observations if not item.invalid_for_scientific_synthesis]
    candidates: list[NextRoundCandidate] = []
    for index, observation in enumerate(scientific[:3], start=1):
        if observation.observation_type == ObservationType.FALSIFIED:
            title = "区分证伪结果的作用域或机制"
            rationale = "原假设在当前作用域下未获支持；下一轮只能改变一个预声明条件以区分机制，不得原样重跑。"
        elif observation.observation_type == ObservationType.BOUNDARY_CONDITION:
            title = "验证局部边界是否可复现"
            rationale = "结果包含局部条件差异，需在独立种子或时段上复验边界。"
        else:
            title = "独立复验当前实验观察"
            rationale = "已有观察需通过独立种子、时段或数据版本验证稳定性。"
        candidates.append(NextRoundCandidate(
            candidate_id=f"NEXT-{record.experiment_id}-{index}",
            title=title,
            rationale=rationale,
            source_observation_ids=[observation.observation_id],
            source_experiment_ids=[record.experiment_id],
            expected_information_gain=max(0.4, observation.confidence_level - 0.1),
            required_capabilities=[],
            known_risks=["human_approval_required", "scope_must_remain_explicit"],
        ))
    stop_reasons = [] if candidates else ["no_new_scientific_observations"]
    return NextRoundProposalBundle(
        proposal_id=f"NRP-{record.experiment_id}",
        source_experiment_id=record.experiment_id,
        new_observation_ids=[item.observation_id for item in observations],
        candidates=candidates,
        recommended_candidate_id=candidates[0].candidate_id if candidates else None,
        stop_reasons=stop_reasons,
    )


def build_empirical_capability_profile(records: list[HistoricalExperimentRecord]) -> EmpiricalCapabilityProfile:
    models: dict[str, dict[str, Any]] = {}
    for record in records:
        # Planned work, narrative-only legacy mentions, and confirmatory claims
        # whose source artifacts are still unverified do not establish runtime
        # capability.  Counting every candidate name used to turn mentions such
        # as a timed-out TimesNet into a false successful run.
        if record.evidence_tier in {
            EvidenceTier.PLANNED_NOT_EXECUTED,
            EvidenceTier.LEGACY_INFORMATIVE,
        }:
            continue
        if (
            record.evidence_tier == EvidenceTier.AUDITED_CONFIRMATORY
            and record.audit_status != "PASSED"
        ):
            continue
        for model_id in record.candidate_models:
            state = models.setdefault(model_id, {"runs": 0, "success": 0, "failures": 0, "convergence": 0, "runtimes": [], "reasons": []})
            state["runs"] += 1
            if record.evidence_tier == EvidenceTier.ENGINEERING_FAILURE:
                state["failures"] += 1
                state["reasons"].extend(record.known_issues)
                state["convergence"] += sum("converg" in issue.lower() for issue in record.known_issues)
            else:
                state["success"] += 1
            runtime = record.metrics.get("runtime_seconds")
            if isinstance(runtime, (int, float)) and runtime >= 0:
                state["runtimes"].append(float(runtime))
    entries = []
    for model_id, state in sorted(models.items()):
        entries.append(ModelCapabilityPerformance(
            model_id=model_id,
            run_count=state["runs"],
            success_count=state["success"],
            failure_count=state["failures"],
            convergence_failure_count=state["convergence"],
            runtime_seconds=state["runtimes"],
            common_failure_reasons=sorted(set(state["reasons"])),
            last_verified_at=datetime.now(timezone.utc),
            confidence=min(1.0, state["runs"] / 5.0),
        ))
    return EmpiricalCapabilityProfile(models=entries)
