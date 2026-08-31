from uuid import uuid4

from boilermind.core.contracts import (
    ResearchProblemSpec,
)


class TestOnlyProblemParser:
    """
    TEST-ONLY problem parser.

    It does not infer scientific meaning from the question.
    Structured fields are explicitly supplied by the test.

    Production research runs must use a real parser,
    such as QwenProblemParser.
    """

    __test__ = False
    is_test_only = True

    def __init__(
        self,
        *,
        research_object: str,
        target_variable: str,
        operating_condition: str,
        manipulated_variables: list[str] | None = None,
        observed_variables: list[str] | None = None,
        context_variables: list[str] | None = None,
        success_criteria: list[str] | None = None,
        constraints: list[str] | None = None,
    ):
        self.research_object = research_object
        self.target_variable = target_variable
        self.operating_condition = operating_condition

        self.manipulated_variables = (
            manipulated_variables or []
        )

        self.observed_variables = (
            observed_variables or []
        )

        self.context_variables = (
            context_variables or []
        )

        self.success_criteria = (
            success_criteria or []
        )

        self.constraints = (
            constraints or []
        )

    def parse(
        self,
        question: str,
    ) -> ResearchProblemSpec:

        question = question.strip()

        if not question:
            raise ValueError(
                "Research question cannot be empty."
            )

        return ResearchProblemSpec(
            problem_id=(
                f"P-{uuid4().hex[:12]}"
            ),

            original_question=question,

            research_object=(
                self.research_object
            ),

            target_variable=(
                self.target_variable
            ),

            operating_condition=(
                self.operating_condition
            ),

            manipulated_variables=list(
                self.manipulated_variables
            ),

            observed_variables=list(
                self.observed_variables
            ),

            context_variables=list(
                self.context_variables
            ),

            # Important:
            # the research goal comes from the user's
            # question, not from a hard-coded boiler task.
            research_goal=question,

            success_criteria=list(
                self.success_criteria
            ),

            constraints=list(
                self.constraints
            ),
        )