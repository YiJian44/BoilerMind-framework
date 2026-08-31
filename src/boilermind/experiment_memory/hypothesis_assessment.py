from __future__ import annotations

import re
from typing import Any

from boilermind.core.contracts import ExperimentMemoryBundle

from .opportunity import check_hypothesis_duplication


def _tokens(value: Any) -> set[str]:
    text = str(value).casefold()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        for width in (2, 3, 4):
            tokens.update(
                segment[index:index + width]
                for index in range(max(0, len(segment) - width + 1))
            )
    return tokens


def _hypothesis_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(key, "")) for key in (
        "title", "hypothesis_statement", "engineering_mechanism",
        "expected_observation", "key_variables", "applicability_conditions",
    ))


def _scope_mismatches(observation: Any, problem: dict[str, Any]) -> list[str]:
    scope = observation.scope_signature
    mismatches: list[str] = []
    target = str(problem.get("target_variable", "")).strip().casefold()
    if target and scope.target_variable and target != scope.target_variable.casefold():
        mismatches.append("target_variable")
    horizon = problem.get("required_horizon_steps")
    if (
        horizon is not None
        and scope.prediction_horizon_steps is not None
        and int(horizon) != int(scope.prediction_horizon_steps)
    ):
        mismatches.append("prediction_horizon_steps")
    return mismatches


def assess_hypotheses_with_memory(
    hypotheses: list[dict[str, Any]],
    memory: ExperimentMemoryBundle | dict[str, Any],
    problem: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach hypothesis-specific history without modifying LLM semantics."""
    bundle = memory if isinstance(memory, ExperimentMemoryBundle) else ExperimentMemoryBundle.model_validate(memory)
    observations = [
        *bundle.supported_observations,
        *bundle.falsified_observations,
        *bundle.contradictions,
        *bundle.engineering_failures,
    ]
    supported_ids = {item.observation_id for item in bundle.supported_observations}
    conflicting_ids = {item.observation_id for item in [*bundle.falsified_observations, *bundle.contradictions]}
    engineering_ids = {item.observation_id for item in bundle.engineering_failures}
    enriched: list[dict[str, Any]] = []

    for original in hypotheses:
        item = dict(original)
        query_tokens = _tokens(_hypothesis_text(item))
        direct: list[str] = []
        conflicts: list[str] = []
        conditional: list[str] = []
        scope_mismatches: list[dict[str, Any]] = []
        experiment_ids: set[str] = set()

        for observation in observations:
            claim_tokens = _tokens(observation.claim)
            overlap = len(query_tokens & claim_tokens) / max(
                len(query_tokens | claim_tokens), 1
            )
            if overlap < 0.08:
                continue
            mismatches = _scope_mismatches(observation, problem)
            if mismatches:
                scope_mismatches.append({"observation_id": observation.observation_id, "fields": mismatches})
                continue
            observation_model = str(
                observation.supporting_metrics.get("model", "")
            ).strip().casefold()
            hypothesis_text = _hypothesis_text(item).casefold()
            model_specific_history_for_mechanism = (
                bool(observation_model)
                and observation_model not in hypothesis_text
            )
            if (
                observation.observation_id in engineering_ids
                or model_specific_history_for_mechanism
            ):
                conditional.append(observation.observation_id)
                continue
            experiment_ids.update(observation.source_experiment_ids)
            if observation.observation_id in conflicting_ids:
                conflicts.append(observation.observation_id)
            elif observation.observation_id in supported_ids:
                direct.append(observation.observation_id)
            else:
                conditional.append(observation.observation_id)

        duplicate = check_hypothesis_duplication(item, bundle)
        duplicate_ids = [duplicate["duplicate_of"]] if duplicate.get("duplicate") else []
        support_level = (
            "MIXED" if direct and conflicts else
            "STRONG" if len(direct) >= 2 else
            "WEAK" if direct else "NONE"
        )
        duplicate_status = (
            "DUPLICATE" if duplicate_ids else
            "PARTIAL" if float(duplicate.get("similarity", 0.0)) >= 0.45 else
            "NEW"
        )
        evidence_gap = list(item.get("evidence_needed", []))
        if not direct and not conflicts:
            evidence_gap.append("没有检索到同作用域历史实验")

        item["historical_assessment"] = {
            "directly_supporting_observations": direct,
            "conflicting_observations": conflicts,
            "conditionally_related_observations": conditional,
            "duplicate_experiment_ids": duplicate_ids,
            "scope_mismatches": scope_mismatches,
            "historical_support_level": support_level,
            "duplicate_status": duplicate_status,
            "evidence_gap": list(dict.fromkeys(evidence_gap)),
        }
        item["duplicate_check"] = duplicate
        # 记忆匹配无新结果时保留假设已有 grounding（如数据属性画像工厂注入的
        # 历史观测/实验 id），避免覆盖清空导致 no_empirical_grounding。
        new_obs = list(dict.fromkeys([*direct, *conflicts, *conditional]))
        new_exp = sorted(experiment_ids)
        item["source_observation_ids"] = (
            new_obs or list(item.get("source_observation_ids") or [])
        )
        item["source_experiment_ids"] = (
            new_exp or list(item.get("source_experiment_ids") or [])
        )
        item["trigger_types"] = (
            ["HUMAN_PROPOSAL", "HISTORICAL_EXPERIMENT"]
            if item["source_observation_ids"] else ["HUMAN_PROPOSAL"]
        )
        item["workflow_status"] = "HISTORY_ASSESSED"
        enriched.append(item)
    return enriched
