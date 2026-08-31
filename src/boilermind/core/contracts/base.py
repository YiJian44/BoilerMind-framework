from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base model for all BoilerMind scientific contracts."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )