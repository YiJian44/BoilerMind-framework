from boilermind.core.contracts import (
    ScientificHypothesis,
    ScientificResult,
)

from .contracts import KnowledgeExtraction


class TestOnlyKnowledgeExtractor:
    """
    TEST-ONLY deterministic knowledge extractor.

    It simulates the future Qwen entity/relation
    extraction interface.

    It must not be treated as formal Qwen extraction.
    """

    __test__ = False

    is_test_only = True

    def __init__(
        self,
        extractions: dict[
            str,
            KnowledgeExtraction,
        ],
    ):
        self._extractions = extractions

    def extract(
        self,
        hypothesis: ScientificHypothesis,
        scientific_result: ScientificResult,
    ) -> KnowledgeExtraction:

        if (
            hypothesis.hypothesis_id
            != scientific_result.hypothesis_id
        ):
            raise ValueError(
                "Hypothesis/ScientificResult ID mismatch."
            )

        extraction = self._extractions.get(
            hypothesis.hypothesis_id
        )

        if extraction is None:
            raise ValueError(
                "No TEST-ONLY knowledge extraction "
                f"configured for {hypothesis.hypothesis_id}"
            )

        if (
            extraction.hypothesis_id
            != hypothesis.hypothesis_id
        ):
            raise ValueError(
                "Knowledge extraction hypothesis "
                "ID mismatch."
            )

        return extraction