from typing import Protocol

from boilermind.core.contracts import (
    ResearchProblemSpec,
)


class ResearchProblemParser(Protocol):
    """
    Convert a user's natural-language research question
    into a structured ResearchProblemSpec.
    """

    def parse(
        self,
        question: str,
    ) -> ResearchProblemSpec:
        ...