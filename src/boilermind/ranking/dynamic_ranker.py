from __future__ import annotations

from pydantic import Field, model_validator

from boilermind.core.contracts.base import ContractModel
from boilermind.core.enums import ScientificVerdict


class ExperimentFeedback(ContractModel):
    hypothesis_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    verdict: ScientificVerdict
    priority_delta: float = Field(ge=-1.0, le=1.0)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_delta_direction(self):
        if self.verdict == ScientificVerdict.FALSIFIED and self.priority_delta > 0:
            raise ValueError("falsified_feedback_must_not_be_positive")
        if self.verdict == ScientificVerdict.SUPPORTED and self.priority_delta < 0:
            raise ValueError("supported_feedback_must_not_be_negative")
        return self


RELATION_FACTORS = {
    "DIRECT_SUPPORT": 0.80,
    "SHARED_PREMISE_SUPPORT": 0.40,
    "CONDITIONALLY_RELATED": 0.15,
    "SHARED_PREMISE_CONFLICT": 0.40,
    "UNRELATED": 0.0,
}


def dynamic_score(prior_score: float, cumulative_feedback: float) -> float:
    prior = max(0.0, min(1.0, float(prior_score)))
    updated = max(0.0, min(1.0, prior + float(cumulative_feedback)))
    return round(0.65 * prior + 0.35 * updated, 6)
