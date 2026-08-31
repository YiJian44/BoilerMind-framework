from datetime import datetime

from pydantic import Field

from boilermind.core.enums import (
    ExperimentStatus,
    ScientificVerdict,
)

from .base import ContractModel


class ExperimentPlan(ContractModel):
    plan_id: str = Field(min_length=1)

    hypothesis_id: str = Field(min_length=1)

    problem_id: str | None = None

    hypothesis_statement: str = ""
    hypothesis_binding: dict = Field(default_factory=dict)

    experiment_type: str = "model_comparison"

    required_operations: list[str] = Field(
        default_factory=list,
    )

    candidate_models: list[str] = Field(
        default_factory=list,
    )

    recommended_models: list[str] = Field(
        default_factory=list,
    )

    executable_models: list[str] = Field(
        default_factory=list,
    )

    model_substitution_reason: str = ""

    reference_models: list[str] = Field(
        default_factory=list,
    )

    control: dict = Field(default_factory=dict)

    treatment: dict = Field(default_factory=dict)

    target: str | None = None

    target_inference_reason: str = ""

    prediction_horizon_steps: int | None = None

    primary_metric: str | None = None

    secondary_metrics: list[str] = Field(
        default_factory=list,
    )

    hard_constraints: list[str] = Field(
        default_factory=list,
    )

    execution_requirements: dict = Field(default_factory=dict)
    allow_partial_failure: bool = False
    max_runtime_per_model: float | None = None
    max_epochs: int | None = None
    allowed_devices: list[str] = Field(default_factory=lambda: ["cpu", "cuda"])
    reuse_checkpoint_models: list[str] = Field(default_factory=list)

    current_executable: bool = False

    missing_capabilities: list[str] = Field(
        default_factory=list,
    )

    model_selection_rationale: str = ""

    # ---- runtime execution fields (real backend contract) ----
    execution_backend: str = "real_sklearn"

    dataset_path: str = ""

    window_steps: int = 20

    sampling_interval_seconds: int = 15

    train_ratio: float = 0.70

    validation_ratio: float = 0.15

    model_candidates: list[str] = Field(
        default_factory=list,
    )

    reference_model: str | None = None

    selection_metric: str = ""

    locked_test_used_for_selection: bool = False

    random_seed: int = 42

    output_dir: str = ""

    status: str = "planned"

    objective: str = Field(min_length=1)

    experimental_design: str = Field(min_length=1)

    baseline_description: str = Field(min_length=1)

    intervention_description: str = Field(min_length=1)

    required_variables: list[str] = Field(min_length=1)

    metrics: list[str] = Field(min_length=1)

    expected_observation: str = Field(min_length=1)

    confirmation_criteria: list[str] = Field(min_length=1)

    falsification_criteria: list[str] = Field(min_length=1)


class ExperimentContract(ContractModel):
    experiment_id: str = Field(min_length=1)

    problem_id: str | None = None

    hypothesis_id: str = Field(min_length=1)

    plan_id: str = Field(min_length=1)
    hypothesis_binding: dict = Field(default_factory=dict)

    experiment_type: str | None = None

    control: dict = Field(default_factory=dict)

    treatment: dict = Field(default_factory=dict)

    primary_metric: str | None = None

    secondary_metrics: list[str] = Field(
        default_factory=list,
    )

    reference_models: list[str] = Field(
        default_factory=list,
    )

    model_selection_rationale: str = ""

    recommended_models: list[str] = Field(
        default_factory=list,
    )

    executable_models: list[str] = Field(
        default_factory=list,
    )

    model_substitution_reason: str = ""

    target_inference_reason: str = ""

    prediction_horizon_steps: int | None = None

    sampling_interval_seconds: int | None = None

    window_steps: int | None = None

    locked_test_used_for_selection: bool = False

    required_operations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    execution_requirements: dict = Field(default_factory=dict)
    allow_partial_failure: bool = False
    max_runtime_per_model: float | None = None
    max_epochs: int | None = None
    allowed_devices: list[str] = Field(default_factory=lambda: ["cpu", "cuda"])
    reuse_checkpoint_models: list[str] = Field(default_factory=list)

    # Optional, post-experiment provenance for a deliberately created next
    # contract.  The runner does not consume or calculate this suggestion.
    optimization_suggestion: dict | None = None

    dataset_id: str = Field(min_length=1)

    dataset_hash: str = Field(min_length=1)

    input_variables: list[str] = Field(min_length=1)

    target_variable: str = Field(min_length=1)

    train_split: str = Field(min_length=1)

    validation_split: str = Field(min_length=1)

    test_split: str = Field(min_length=1)

    baseline_models: list[str] = Field(min_length=1)

    candidate_models: list[str] = Field(min_length=1)

    metrics: list[str] = Field(min_length=1)

    confirmation_criteria: list[str] = Field(min_length=1)

    falsification_criteria: list[str] = Field(min_length=1)

    random_seed: int = 42

    status: ExperimentStatus = ExperimentStatus.PLANNED


class ExperimentResult(ContractModel):
    experiment_id: str = Field(min_length=1)

    problem_id: str | None = None

    hypothesis_id: str = Field(min_length=1)

    plan_id: str | None = None

    status: ExperimentStatus

    metrics: dict[str, float] = Field(default_factory=dict)

    # The runner preserves the source metric keys and publishes a separate
    # canonical view. Audit consumes only normalized_metrics.
    raw_metrics: dict[str, float] = Field(default_factory=dict)

    normalized_metrics: dict[str, float | str] = Field(default_factory=dict)

    baseline_metrics: dict[str, float] = Field(default_factory=dict)

    # Per-model locked-test metrics for every predeclared
    # candidate and the reference model (e.g. persistence).
    # This is what hypothesis-level verdicts must read,
    # NOT the single validation-selected model only.
    candidate_locked_test_metrics: dict[
        str,
        dict[str, float],
    ] = Field(default_factory=dict)

    # Standardized per-model execution records.
    model_records: dict[
        str,
        "ModelExperimentRecord",
    ] = Field(default_factory=dict)

    artifacts: list[str] = Field(default_factory=list)

    execution_notes: list[str] = Field(default_factory=list)

    control_metrics: dict[str, float] = Field(default_factory=dict)
    treatment_metrics: dict[str, float] = Field(default_factory=dict)
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    regime_metrics: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )
    conclusion_scope: str = "full_hypothesis"
    experiment_valid: bool | None = None
    experiment_validity_issues: list[str] = Field(default_factory=list)

    started_at: datetime

    completed_at: datetime | None = None


class ModelExperimentRecord(ContractModel):
    """
    Standardized record for one model inside one experiment.

    Warnings / convergence are explicit fields - a model that
    produced a ConvergenceWarning must not look identical to a
    fully converged model. A failed model keeps failure_reason;
    it is never silently replaced by another model.
    """

    model_name: str = Field(min_length=1)

    fit_success: bool

    fit_converged: bool

    warnings: list[str] = Field(
        default_factory=list,
    )

    failure_reason: str | None = None

    runtime_seconds: float | None = None

    model_configuration: dict = Field(
        default_factory=dict,
        serialization_alias="model_config",
    )

    validation_metrics: dict[str, float] = Field(
        default_factory=dict,
    )

    locked_test_metrics: dict[str, float] = Field(
        default_factory=dict,
    )

    train_samples: int | None = None

    validation_samples: int | None = None

    test_samples: int | None = None

    random_seed: int | None = None

    dataset_sha256: str | None = None

    artifact_paths: list[str] = Field(
        default_factory=list,
    )

    artifact_provenance: dict = Field(default_factory=dict)

    device: str | None = None
    epochs_completed: int | None = None
    best_epoch: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None


class ExperimentAudit(ContractModel):
    experiment_id: str = Field(min_length=1)

    execution_valid: bool

    dataset_frozen: bool

    leakage_check_passed: bool

    baseline_valid: bool

    metric_check_passed: bool

    issues: list[str] = Field(default_factory=list)


class ScientificResult(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    experiment_id: str = Field(min_length=1)

    verdict: ScientificVerdict

    rationale: str = Field(min_length=1)

    achieved_criteria: list[str] = Field(default_factory=list)

    failed_criteria: list[str] = Field(default_factory=list)
