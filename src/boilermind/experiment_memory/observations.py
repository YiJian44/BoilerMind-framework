from __future__ import annotations

import hashlib
import re

from boilermind.core.contracts import (
    EvidenceTier,
    ExperimentObservation,
    HistoricalExperimentRecord,
    ObservationType,
)


def _observation_id(experiment_id: str, claim: str, index: int) -> str:
    digest = hashlib.sha256(f"{experiment_id}|{index}|{claim}".encode("utf-8")).hexdigest()[:12]
    return f"OBS-{digest}"


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\*\*|`", "", text or "")
    return [part.strip(" -。；;\n") for part in re.split(r"[。；;\n]+", cleaned) if part.strip(" -。；;\n")]


def _kind(claim: str, record: HistoricalExperimentRecord) -> ObservationType:
    lowered = claim.lower()
    # Lack of evidence is neither support nor falsification.  Check this
    # before the broad legacy "fail" token rule, otherwise phrases such as
    # "evidence is insufficient or audit failed" poison memory as FALSIFIED.
    if any(word in lowered for word in (
        "insufficient_evidence", "insufficient evidence",
        "证据不足", "cannot provide valid ranking feedback",
        "无法提供有效排序反馈",
    )):
        return ObservationType.PARTIAL
    if record.evidence_tier == EvidenceTier.ENGINEERING_FAILURE or any(word in lowered for word in ("bug", "超时", "错位", "泄漏", "失败原因")):
        return ObservationType.ENGINEERING_FAILURE
    if any(word in lowered for word in ("不一致", "样本量较小", "需重跑", "未对齐", "缺失")) or re.search(r"仅\s*\d+\s*样本", claim):
        return ObservationType.DATA_QUALITY_WARNING
    if "✅" in claim and ("fail" in lowered or "❌" in claim):
        return ObservationType.PARTIAL
    if any(word in lowered for word in ("证伪", "不支持", "被拒", "fail", "未达", "打不过", "劣于", "差于")) or "❌" in claim:
        return ObservationType.FALSIFIED
    if any(word in lowered for word in ("仅", "局部", "但", "条件", "工况", "不迁移")):
        return ObservationType.BOUNDARY_CONDITION
    if any(word in lowered for word in ("支持", "pass", "成立", "优于", "全赢", "最优")) or "✅" in claim:
        return ObservationType.SUPPORTED
    return ObservationType.PARTIAL


def derive_experiment_observations(record: HistoricalExperimentRecord) -> list[ExperimentObservation]:
    if record.evidence_tier == EvidenceTier.PLANNED_NOT_EXECUTED:
        return []
    # A runtime outcome is already structured and audited. Splitting its
    # machine-formatted rationale used to create fake standalone observations
    # such as ``metrics=...`` and ``better=ridge``. Preserve one scientific
    # observation per executed experiment instead.
    if record.source_type == "boilermind_runtime":
        verdict = str(record.verdict or "INSUFFICIENT_EVIDENCE").upper()
        audit_valid = (
            record.evidence_tier != EvidenceTier.ENGINEERING_FAILURE
            and record.audit_status == "PASSED"
        )
        observation_type = {
            "SUPPORTED": ObservationType.SUPPORTED,
            "PARTIALLY_SUPPORTED": ObservationType.PARTIAL,
            "FALSIFIED": ObservationType.FALSIFIED,
        }.get(verdict, ObservationType.PARTIAL)
        if not audit_valid:
            observation_type = ObservationType.ENGINEERING_FAILURE
        claim = (
            f"实验 {record.experiment_id} 在冻结作用域内的科学判定为 {verdict}。"
        )
        return [ExperimentObservation(
            observation_id=_observation_id(record.experiment_id, claim, 1),
            source_experiment_ids=[record.experiment_id],
            observation_type=observation_type,
            claim=claim,
            scope_signature=record.scope,
            comparison_signature=record.scope.model_dump_json(exclude_none=True),
            supporting_metrics=dict(record.metrics),
            counter_evidence=list(record.known_issues),
            confidence_level=0.9 if audit_valid else 0.0,
            reuse_policy=(
                "SCOPE_MATCH_REQUIRED" if audit_valid else "ENGINEERING_ONLY"
            ),
            invalid_for_scientific_synthesis=not audit_valid,
            derivation_version="2.0.0",
        )]
    candidates = _sentences(record.raw_result) + _sentences(record.raw_limitations)
    observations: list[ExperimentObservation] = []
    seen: set[str] = set()
    base_confidence = {
        EvidenceTier.AUDITED_CONFIRMATORY: 0.9,
        EvidenceTier.AUDITED_EXPLORATORY: 0.65,
        EvidenceTier.LEGACY_INFORMATIVE: 0.4,
        EvidenceTier.ENGINEERING_FAILURE: 0.8,
    }.get(record.evidence_tier, 0.2)
    if record.evidence_tier == EvidenceTier.AUDITED_CONFIRMATORY and record.audit_status != "PASSED":
        base_confidence = 0.55
    for index, claim in enumerate(candidates, start=1):
        if claim.startswith("执行工具"):
            continue
        normalized = re.sub(r"\s+", "", claim.lower())
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        kind = _kind(claim, record)
        invalid = kind == ObservationType.ENGINEERING_FAILURE or record.evidence_tier == EvidenceTier.ENGINEERING_FAILURE
        reuse_policy = "ENGINEERING_ONLY" if invalid else "SCOPE_MATCH_REQUIRED"
        if record.evidence_tier == EvidenceTier.AUDITED_CONFIRMATORY and record.audit_status != "PASSED":
            reuse_policy = "SOURCE_ARTIFACT_VERIFICATION_REQUIRED"
        observations.append(ExperimentObservation(
            observation_id=_observation_id(record.experiment_id, claim, index),
            source_experiment_ids=[record.experiment_id],
            observation_type=kind,
            claim=claim,
            scope_signature=record.scope,
            comparison_signature=record.scope.model_dump_json(exclude_none=True),
            confidence_level=base_confidence,
            reuse_policy=reuse_policy,
            invalid_for_scientific_synthesis=invalid,
        ))
    return observations
