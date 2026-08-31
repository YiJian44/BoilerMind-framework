from typing import Literal

from pydantic import Field

from .base import ContractModel


class ResearchProblemSpec(ContractModel):
    problem_id: str = Field(min_length=1)

    original_question: str = Field(min_length=1)

    research_object: str = Field(min_length=1)

    target_variable: str = Field(min_length=1)

    raw_target_variable: str | None = None

    normalized_target_variable: str | None = None

    target_normalization_reason: str | None = None

    # The physical/engineering quantity being predicted is kept
    # separate from the experiment's evaluation objective.
    objective: str = Field(default="unspecified", min_length=1)

    metrics: list[str] = Field(default_factory=list)

    target_inference_reason: str = ""

    operating_condition: str = Field(min_length=1)

    manipulated_variables: list[str] = Field(default_factory=list)

    observed_variables: list[str] = Field(default_factory=list)

    context_variables: list[str] = Field(default_factory=list)

    research_goal: str = Field(min_length=1)

    success_criteria: list[str] = Field(default_factory=list)

    constraints: list[str] = Field(default_factory=list)

    # Deterministically extracted execution constraints from the original
    # question.  These are not delegated to the LLM and must survive into the
    # experiment plan.
    required_models: list[str] = Field(default_factory=list)
    reference_models: list[str] = Field(default_factory=list)
    required_horizon_steps: int | None = Field(default=None, ge=1)
    required_operations: list[str] = Field(default_factory=list)
    protocol_constraints: list[str] = Field(default_factory=list)
    required_objective_dimensions: list[str] = Field(default_factory=list)

    # Backward-compatible, deterministic research-task routing metadata.
    research_task_type: Literal[
        "hypothesis_validation",
        "parameter_optimization",
        "model_selection",
        "feature_analysis",
    ] = "hypothesis_validation"
    optimization_variable: str | None = None
    candidate_values: list[int | float | str] = Field(default_factory=list)
    task_type_resolution_reason: str = ""
