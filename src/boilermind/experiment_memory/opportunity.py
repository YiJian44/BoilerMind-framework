from __future__ import annotations

import hashlib
import re
from typing import Any

from boilermind.core.contracts import (
    CurrentObservationBundle,
    ExperimentMemoryBundle,
    HypothesisTrigger,
    Opportunity,
    OpportunityMap,
)


def _id(problem_id: str, category: str, sources: list[str]) -> str:
    digest = hashlib.sha256(f"{problem_id}|{category}|{'|'.join(sorted(sources))}".encode("utf-8")).hexdigest()[:12]
    return f"OPP-{digest}"


def build_opportunity_map(
    memory: ExperimentMemoryBundle,
    current_observations: CurrentObservationBundle | None,
    capability: dict[str, Any],
) -> OpportunityMap:
    opportunities: list[Opportunity] = []
    executable = bool(capability.get("enabled_experiment_models"))

    if memory.contradictions:
        source_ids = [item.observation_id for item in memory.contradictions[:4]]
        experiment_ids = sorted({eid for item in memory.contradictions[:4] for eid in item.source_experiment_ids})
        opportunities.append(Opportunity(
            opportunity_id=_id(memory.problem_id, "SCOPE_BOUNDARY", source_ids),
            category="SCOPE_BOUNDARY",
            title="验证历史结果的作用域边界",
            rationale="历史观察显示整体与局部工况、目标口径或窗口条件下的结果可能不同，应设计区分边界的实验。",
            trigger_types=[HypothesisTrigger.HISTORICAL_EXPERIMENT, HypothesisTrigger.CONTRADICTORY_RESULTS],
            source_observation_ids=source_ids,
            source_experiment_ids=experiment_ids,
            expected_information_gain=0.9,
            currently_executable=executable,
            missing_capabilities=[] if executable else ["enabled_experiment_models"],
            do_not_repeat_experiment_ids=experiment_ids,
        ))

    if memory.falsified_observations:
        source_ids = [item.observation_id for item in memory.falsified_observations[:4]]
        experiment_ids = sorted({eid for item in memory.falsified_observations[:4] for eid in item.source_experiment_ids})
        opportunities.append(Opportunity(
            opportunity_id=_id(memory.problem_id, "FAILURE_DISCRIMINATION", source_ids),
            category="FAILURE_DISCRIMINATION",
            title="区分历史证伪的机制原因与适用范围",
            rationale="不得重复同条件失败实验；下一假设应改变一个可解释条件以区分机制或确认不迁移边界。",
            trigger_types=[HypothesisTrigger.HISTORICAL_EXPERIMENT],
            source_observation_ids=source_ids,
            source_experiment_ids=experiment_ids,
            expected_information_gain=0.75,
            currently_executable=executable,
            missing_capabilities=[] if executable else ["enabled_experiment_models"],
            do_not_repeat_experiment_ids=experiment_ids,
        ))

    if memory.supported_observations:
        source_ids = [item.observation_id for item in memory.supported_observations[:3]]
        experiment_ids = sorted({eid for item in memory.supported_observations[:3] for eid in item.source_experiment_ids})
        opportunities.append(Opportunity(
            opportunity_id=_id(memory.problem_id, "REPLICATION", source_ids),
            category="REPLICATION",
            title="在独立种子、时段或数据版本上复验已有支持",
            rationale="已有支持只能在原作用域内复用；独立复验可检验稳定性而不是重复堆叠同一记录。",
            trigger_types=[HypothesisTrigger.HISTORICAL_EXPERIMENT],
            source_observation_ids=source_ids,
            source_experiment_ids=experiment_ids,
            expected_information_gain=0.65,
            currently_executable=executable,
            missing_capabilities=[] if executable else ["enabled_experiment_models"],
            do_not_repeat_experiment_ids=experiment_ids,
        ))

    if memory.engineering_failures:
        source_ids = [item.observation_id for item in memory.engineering_failures[:3]]
        experiment_ids = sorted({eid for item in memory.engineering_failures[:3] for eid in item.source_experiment_ids})
        opportunities.append(Opportunity(
            opportunity_id=_id(memory.problem_id, "PROTOCOL_REPAIR", source_ids),
            category="PROTOCOL_REPAIR",
            title="修复历史协议或执行缺陷后重跑",
            rationale="工程失败不能作为科学证伪；只有修复已记录缺陷并保持其他条件冻结后才能重新评价。",
            trigger_types=[HypothesisTrigger.HISTORICAL_EXPERIMENT, HypothesisTrigger.CAPABILITY_EXPANSION],
            source_observation_ids=source_ids,
            source_experiment_ids=experiment_ids,
            expected_information_gain=0.8,
            currently_executable=False,
            missing_capabilities=["verified_protocol_repair"],
            do_not_repeat_experiment_ids=experiment_ids,
        ))

    if current_observations and current_observations.observations:
        opportunity_sources = [str(item.get("observation_id", index)) for index, item in enumerate(current_observations.observations)]
        opportunities.append(Opportunity(
            opportunity_id=_id(memory.problem_id, "CURRENT_DATA", opportunity_sources),
            category="CURRENT_DATA",
            title="验证当前数据观察是否稳定",
            rationale="当前数据观察尚不是科学结论，应使用锁定测试和预声明标准进行验证。",
            trigger_types=[HypothesisTrigger.CURRENT_DATA_OBSERVATION],
            expected_information_gain=0.85,
            currently_executable=executable,
            missing_capabilities=[] if executable else ["enabled_experiment_models"],
        ))

    opportunities.sort(key=lambda item: (-item.expected_information_gain, item.opportunity_id))
    stop_reasons = []
    if not opportunities:
        stop_reasons.append("no_qualified_memory_or_data_opportunity")
    if opportunities and not any(item.currently_executable for item in opportunities):
        stop_reasons.append("only_unexecutable_opportunities_remain")
    return OpportunityMap(problem_id=memory.problem_id, opportunities=opportunities[:10], stop_reasons=stop_reasons)


def check_hypothesis_duplication(candidate: dict[str, Any], memory: ExperimentMemoryBundle) -> dict[str, Any]:
    text = " ".join(str(candidate.get(key, "")) for key in ("title", "hypothesis", "verification_intent")).lower()
    candidate_tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text))
    best_id = None
    best_score = 0.0
    for observation in [*memory.supported_observations, *memory.falsified_observations, *memory.contradictions]:
        tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", observation.claim.lower()))
        score = len(candidate_tokens & tokens) / max(len(candidate_tokens | tokens), 1)
        if score > best_score:
            best_score = score
            best_id = observation.source_experiment_ids[0]
    return {"duplicate": best_score >= 0.78, "duplicate_of": best_id if best_score >= 0.78 else None, "similarity": round(best_score, 4)}
