from pydantic import Field

from boilermind.core.contracts.base import ContractModel


class ExperimentCapabilitySnapshot(ContractModel):
    """
    Frozen snapshot of what BoilerMind can actually execute.
    """

    snapshot_id: str = Field(min_length=1)

    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)

    available_variables: list[str] = Field(
        min_length=1
    )

    available_target_variables: list[str] = Field(
        min_length=1
    )

    available_baseline_models: list[str] = Field(
        min_length=1
    )

    available_candidate_models: list[str] = Field(
        min_length=1
    )

    available_metrics: list[str] = Field(
        min_length=1
    )

    train_split: str = Field(min_length=1)
    validation_split: str = Field(min_length=1)
    test_split: str = Field(min_length=1)

    data_frozen: bool
    leakage_policy_verified: bool

    dataset_path: str | None = None

    prediction_horizon_steps: int | None = None

    sampling_interval_seconds: int | None = None


class PlanCritiqueDecision(ContractModel):
    """
    Structured scientific review of an experiment plan.

    The reviewer may later be Qwen + deterministic rules,
    but final admission is controlled by Python.
    """

    plan_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)

    hypothesis_experiment_alignment: bool

    intervention_valid: bool

    baseline_valid: bool

    metric_alignment: bool

    confirmation_falsification_valid: bool

    executable: bool

    issues: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(min_length=1)


class PlanApprovalReport(ContractModel):
    plan_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)

    passed: bool

    issues: list[str] = Field(
        default_factory=list
    )

    experiment_id: str | None = None
