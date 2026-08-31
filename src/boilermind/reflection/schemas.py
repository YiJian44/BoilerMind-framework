from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


WINDOW_SIZES = (5, 10, 20, 40, 60)
FORECAST_HORIZONS = (1, 5, 10)
MODEL_WIDTHS = (32, 64, 128)
LEARNING_RATES = (1e-4, 5e-4, 1e-3, 5e-3)


class WhitelistedConfiguration(BaseModel):
    """Only parameters explicitly approved for deterministic suggestions."""

    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None
    window_size: Literal[5, 10, 20, 40, 60] | None = None
    forecast_horizon: Literal[1, 5, 10] | None = None
    d_model: Literal[32, 64, 128] | None = None
    hidden_size: Literal[32, 64, 128] | None = None
    learning_rate: Literal[0.0001, 0.0005, 0.001, 0.005] | None = None

    @model_validator(mode="after")
    def validate_model_specific_parameters(self):
        model = (self.model_name or "").casefold()
        if self.d_model is not None and model != "transformer":
            raise ValueError("d_model_is_only_valid_for_transformer")
        if self.hidden_size is not None and model not in {"lstm", "gru"}:
            raise ValueError("hidden_size_is_only_valid_for_lstm_or_gru")
        return self


class PerformanceAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "stable",
        "high_error_relative_to_baseline",
        "not_comparable",
        "execution_failed",
    ]
    metric: str | None = None
    observed_value: float | None = None
    baseline_value: float | None = None
    relative_change: float | None = None
    evidence: list[str] = Field(default_factory=list)


class ExperimentOptimizationSuggestion(BaseModel):
    """A suggestion derived from existing facts, never an experiment result."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    current_configuration: dict[str, Any]
    performance_analysis: PerformanceAnalysis
    next_configuration: WhitelistedConfiguration
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    changed_parameters: list[str] = Field(default_factory=list)
