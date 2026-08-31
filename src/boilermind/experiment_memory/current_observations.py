from __future__ import annotations

import hashlib
from typing import Any

from boilermind.core.contracts import CurrentObservationBundle, ResearchProblemSpec


def extract_current_observations(
    problem: ResearchProblemSpec,
    capability: dict[str, Any],
) -> CurrentObservationBundle:
    """Expose verified runtime/data facts without turning them into scientific conclusions."""
    dataset = capability.get("dataset_contract") or capability.get("dataset") or {}
    observations: list[dict[str, Any]] = []

    def add(kind: str, fact: str, payload: dict[str, Any]) -> None:
        digest = hashlib.sha256(f"{problem.problem_id}|{kind}|{fact}".encode("utf-8")).hexdigest()[:12]
        observations.append({
            "observation_id": f"CUR-{digest}",
            "observation_type": kind,
            "fact": fact,
            "payload": payload,
            "scientific_conclusion": False,
            "reuse_policy": "RUNTIME_FACT_ONLY",
        })

    if dataset:
        add("DATASET_CAPABILITY", "当前运行时存在可执行数据合同。", {
            "dataset_id": dataset.get("dataset_id") or dataset.get("id"),
            "dataset_hash": dataset.get("dataset_hash") or dataset.get("sha256"),
            "row_count": dataset.get("row_count"),
            "target_variable": dataset.get("target_variable"),
        })
    models = list(capability.get("enabled_experiment_models") or [])
    if models:
        add("MODEL_CAPABILITY", "当前运行时已验证一组可执行模型。", {"enabled_models": models})
    operations = list(capability.get("supported_experiment_operations") or [])
    if operations:
        add("OPERATION_CAPABILITY", "当前运行时只允许声明过的实验操作。", {"supported_operations": operations})
    return CurrentObservationBundle(problem_id=problem.problem_id, observations=observations)
