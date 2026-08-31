from typing import Protocol

from boilermind.core.contracts import (
    ScientificHypothesis,
    ScientificResult,
)

from .contracts import KnowledgeExtraction


class KnowledgeExtractor(Protocol):

    def extract(
        self,
        hypothesis: ScientificHypothesis,
        scientific_result: ScientificResult,
    ) -> KnowledgeExtraction:
        ...