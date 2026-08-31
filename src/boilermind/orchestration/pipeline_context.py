from datetime import datetime, timezone

from pydantic import Field

from boilermind.core.contracts.base import ContractModel
from boilermind.core.contracts import ResearchProblemSpec


class PipelineContext(ContractModel):
    """
    Persistent context for one BoilerMind research run.
    """

    run_id: str = Field(min_length=1)

    raw_question: str = Field(min_length=1)

    problem: ResearchProblemSpec

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )