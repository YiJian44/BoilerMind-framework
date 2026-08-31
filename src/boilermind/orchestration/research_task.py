from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ResearchTaskType = Literal[
    "hypothesis_validation",
    "parameter_optimization",
    "model_selection",
    "feature_analysis",
]


class ResearchTaskIntent(BaseModel):
    research_task_type: ResearchTaskType
    optimization_variable: str | None = None
    candidate_values: list[int | float | str] = Field(default_factory=list)
    resolution_reason: str


PARAMETER_SPACES: dict[str, tuple[int | float | str, ...]] = {
    "window_steps": (10, 20, 40, 80),
}


def resolve_research_task(question: str) -> ResearchTaskIntent:
    """Resolve experiment semantics deterministically without target assumptions."""
    text = str(question or "").strip().casefold()
    window_terms = (
        "时间窗口", "窗口长度", "历史窗口", "哪个窗口", "最优窗口",
        "window", "history length", "sequence length", "time lag",
    )
    optimization_terms = (
        "哪个", "选择", "最优", "更好", "参数影响", "参数优化",
        "超参数", "optimization", "optimisation", "best", "better",
    )
    if any(term in text for term in window_terms) and any(
        term in text for term in optimization_terms
    ):
        return ResearchTaskIntent(
            research_task_type="parameter_optimization",
            optimization_variable="window_steps",
            candidate_values=list(PARAMETER_SPACES["window_steps"]),
            resolution_reason="deterministic_window_parameter_optimization",
        )
    if any(term in text for term in (
        "模型比较", "哪个模型", "选择模型", "model selection", "compare models",
    )):
        return ResearchTaskIntent(
            research_task_type="model_selection",
            resolution_reason="deterministic_model_selection",
        )
    if any(term in text for term in (
        "特征分析", "变量分析", "特征影响", "feature analysis", "feature importance",
    )):
        return ResearchTaskIntent(
            research_task_type="feature_analysis",
            resolution_reason="deterministic_feature_analysis",
        )
    if any(term in text for term in (
        "参数影响", "参数优化", "超参数优化", "hyperparameter optimization",
        "parameter optimization",
    )):
        return ResearchTaskIntent(
            research_task_type="parameter_optimization",
            resolution_reason="parameter_optimization_variable_unresolved",
        )
    return ResearchTaskIntent(
        research_task_type="hypothesis_validation",
        resolution_reason="default_hypothesis_validation",
    )


def apply_research_task(problem: Any, question: str):
    intent = resolve_research_task(question)
    return problem.model_copy(update={
        "research_task_type": intent.research_task_type,
        "optimization_variable": intent.optimization_variable,
        "candidate_values": list(intent.candidate_values),
        "task_type_resolution_reason": intent.resolution_reason,
    })
