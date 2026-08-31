from __future__ import annotations

from typing import Any

from boilermind.orchestration.qwen_problem_parser import (
    QwenProblemParser,
)
from boilermind.adapters.deterministic_problem_parser import (
    DeterministicProblemParser,
    DeterministicProblemParserError,
)
from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
    ExperimentCapabilityRegistry,
)
from boilermind.orchestration.problem_intake import analyze_problem_intake
from boilermind.orchestration.research_task import apply_research_task

from .base import BaseSkill


class ProblemParsingSkill(BaseSkill):

    name = "problem_parsing"

    description = (
        "使用Qwen将用户自然语言科研问题结构化为"
        "ResearchProblemSpec"
    )

    def __init__(
        self,
        *,
        capability_registry: ExperimentCapabilityRegistry | None = None,
    ):
        self.capability = (
            capability_registry or DirectVolume31VCapabilityRegistry()
        )

    def _safe_defaults(self) -> dict[str, Any]:
        return {
            "candidate_models": self.capability.available_models(),
            "reference_model": self.capability.reference_model_id(),
            "prediction_horizon_steps": (
                self.capability.prediction_horizon_steps_value()
            ),
            "metrics": self.capability.metrics(),
        }

    def execute(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        question = str(
            context.get(
                "research_question",
                "",
            )
        ).strip()

        if not question:
            raise ValueError(
                "research_question_required"
            )

        intake = analyze_problem_intake(question)
        parser = DeterministicProblemParser()
        field_sources: dict[str, str] = {}
        completion_notes: list[str] = []
        try:
            outcome = parser.parse_with_safe_defaults(
                question,
                problem_type=intake.problem_type,
                defaults=self._safe_defaults(),
            )
            problem = outcome.problem
            field_sources = outcome.field_sources
            completion_notes = outcome.completion_notes
            parser_type = (
                "deterministic_supported_question_v2_autocomplete"
                if completion_notes
                else "deterministic_supported_question_v2"
            )
        except DeterministicProblemParserError:
            qwen_parser = QwenProblemParser()
            try:
                problem = qwen_parser.parse(question)
                parser_type = "qwen_ambiguity_fallback"
            finally:
                qwen_parser.close()

        problem = apply_research_task(problem, question)


        problem_payload = (
            problem.model_dump(
                mode="json"
            )
        )


        return {

            # 新的可信科学合同。
            "research_problem":
                problem_payload,

            "problem_id":
                problem.problem_id,

            # 兼容已有 facade / trace。
            "problem_statement":
                question,

            "status":
                "parsed",

            "problem_parser_type": parser_type,
            "field_sources": field_sources,
            "automatic_completions": completion_notes,
            "problem_intake": intake.model_dump(mode="json"),
            "semantic_gaps": list(intake.missing_fields),
            "semantic_assumptions": [
                item.model_dump(mode="json")
                for item in intake.clarification_items
            ],
        }
