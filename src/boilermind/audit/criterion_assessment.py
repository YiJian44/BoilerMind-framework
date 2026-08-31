from pydantic import Field, model_validator

from boilermind.core.contracts.base import ContractModel


class CriterionAssessment(ContractModel):
    experiment_id: str = Field(min_length=1)

    confirmation_met: bool
    falsification_met: bool

    achieved_criteria: list[str] = Field(
        default_factory=list
    )

    failed_criteria: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self):
        if (
            self.confirmation_met
            and self.falsification_met
        ):
            raise ValueError(
                "Confirmation and falsification "
                "cannot both be true."
            )

        return self