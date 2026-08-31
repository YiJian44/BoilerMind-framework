"""Deterministic scientific-knowledge summaries from experiment results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def generate_scientific_knowledge(experiment_result: Mapping[str, Any]) -> str:
    """Generate one scientific summary sentence without calling a language model."""
    hypothesis = str(experiment_result.get("hypothesis", "")).strip()
    condition = str(experiment_result.get("condition", "")).strip()
    if not hypothesis:
        raise ValueError("experiment_result.hypothesis must be non-empty")

    if condition:
        return f"实验结果支持：在{condition}下，{hypothesis}。"
    return f"实验结果支持：{hypothesis}。"
