from pydantic import Field

from boilermind.core.contracts.base import ContractModel


class ExperimentExecutionTrace(ContractModel):
    experiment_id: str = Field(min_length=1)

    dataset_frozen: bool
    leakage_check_passed: bool
    baseline_valid: bool
    metric_check_passed: bool

    notes: list[str] = Field(
        default_factory=list
    )