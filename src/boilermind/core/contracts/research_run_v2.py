from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field

from .base import ContractModel


RunStatusV2 = Literal[
    "QUEUED", "RUNNING", "COMPLETED", "COMPLETED_WITH_REPORT_WARNING",
    "NO_EXECUTABLE_HYPOTHESES", "FAILED",
]


class ResearchRequest(ContractModel):
    question: str = Field(min_length=1)
    run_id: str | None = None


class FieldProvenance(ContractModel):
    field_name: str = Field(min_length=1)
    source: Literal[
        "USER", "DETERMINISTIC", "LLM", "HISTORICAL_EXPERIMENT",
        "CAPABILITY_REGISTRY", "SYSTEM_DEFAULT",
    ]
    value: Any = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class StageTrace(ContractModel):
    stage: str = Field(min_length=1)
    status: Literal["STARTED", "COMPLETED", "FAILED", "SKIPPED"]
    source: Literal["PROGRAM", "LLM", "HYBRID"] = "PROGRAM"
    input_sha256: str | None = None
    output_sha256: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    errors: list[str] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HypothesisScore(ContractModel):
    hypothesis_id: str = Field(min_length=1)
    historical_support: float = Field(ge=0.0, le=1.0)
    historical_scope_match: float = Field(ge=0.0, le=1.0)
    problem_relevance: float = Field(ge=0.0, le=1.0)
    reproducibility: float = Field(ge=0.0, le=1.0)
    falsifiability: float = Field(ge=0.0, le=1.0)
    prior_score: float = Field(ge=0.0, le=1.0)
    cumulative_feedback: float = Field(default=0.0, ge=-1.0, le=1.0)
    dynamic_score: float = Field(ge=0.0, le=1.0)
    eligible: bool = True
    dropped_reasons: list[str] = Field(default_factory=list)


class RankingSnapshot(ContractModel):
    snapshot_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    scoring_version: str = "historical_prior_v2"
    entries: list[HypothesisScore]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HypothesisRunState(ContractModel):
    hypothesis_id: str = Field(min_length=1)
    execution_count: int = Field(default=0, ge=0, le=2)
    eligible: bool = True
    latest_verdict: str | None = None
    cumulative_feedback: float = Field(default=0.0, ge=-1.0, le=1.0)
    executed_design_sha256: list[str] = Field(default_factory=list)
    exit_reason: str | None = None


class BatchMember(ContractModel):
    hypothesis_id: str = Field(min_length=1)
    plan: dict[str, Any]
    contract: dict[str, Any]
    outcome: dict[str, Any] | None = None
    status: Literal["READY", "RUNNING", "COMPLETED", "FAILED"] = "READY"
    issues: list[str] = Field(default_factory=list)


class HypothesisValidationBatch(ContractModel):
    batch_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=3)
    ranking_snapshot_id: str = Field(min_length=1)
    members: list[BatchMember] = Field(min_length=1, max_length=3)
    max_parallelism: Literal[3] = 3
    status: Literal["FROZEN", "RUNNING", "COMPLETED", "FAILED"] = "FROZEN"
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ResearchRunState(ContractModel):
    schema_version: Literal["boilermind.research_run.v2"] = "boilermind.research_run.v2"
    run_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    status: RunStatusV2 = "QUEUED"
    research_problem: dict[str, Any] | None = None
    field_provenance: list[FieldProvenance] = Field(default_factory=list)
    evidence_bundle: dict[str, Any] | None = None
    experiment_memory_bundle: dict[str, Any] | None = None
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    hypothesis_states: dict[str, HypothesisRunState] = Field(default_factory=dict)
    ranking_snapshots: list[RankingSnapshot] = Field(default_factory=list)
    batches: list[HypothesisValidationBatch] = Field(default_factory=list)
    stage_traces: list[StageTrace] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
