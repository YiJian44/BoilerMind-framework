from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from boilermind.core.contracts import ResearchProblemSpec


class DeterministicProblemParserError(ValueError):
    pass


@dataclass(frozen=True)
class DeterministicParseOutcome:
    problem: ResearchProblemSpec
    field_sources: dict[str, str]
    completion_notes: list[str]


class DeterministicProblemParser:
    """Parse only explicit, currently supported model-comparison questions."""

    _MODELS = (
        ("persistence", r"(?<![a-z0-9_])persistence(?![a-z0-9_])|持久性(?:基线)?"),
        ("bayesianridge", r"(?<![a-z0-9_])bayesian\s*ridge(?![a-z0-9_])|(?<![a-z0-9_])bayesianridge(?![a-z0-9_])|贝叶斯岭"),
        ("ridge", r"(?<![a-z0-9_])ridge(?![a-z0-9_])|(?<!贝叶斯)岭回归"),
        ("hgb", r"(?<![a-z0-9_])hgb(?![a-z0-9_])"),
        ("rf", r"\brandom\s*forest\b|\brf\b|随机森林"),
        ("lstm", r"(?<![a-z0-9_])lstm(?![a-z0-9_])"),
        ("transformer", r"(?<![a-z0-9_])transformer(?![a-z0-9_])"),
    )

    @staticmethod
    def _horizon(text: str) -> int | None:
        explicit = re.search(
            r"(?<![a-z0-9_])h\s*(80|40|20|8|4)(?![0-9])", text
        )
        if explicit:
            return int(explicit.group(1))
        step = re.search(r"(80|40|20|8|4)\s*步", text)
        if step:
            return int(step.group(1))
        minute = re.search(r"(20|10|5|2|1)\s*分钟", text)
        if minute:
            return {"20": 80, "10": 40, "5": 20, "2": 8, "1": 4}[
                minute.group(1)
            ]
        return None

    def parse(self, question: str) -> ResearchProblemSpec:
        return self._parse(question, defaults=None).problem

    def parse_with_safe_defaults(
        self,
        question: str,
        *,
        problem_type: str,
        defaults: dict[str, Any],
    ) -> DeterministicParseOutcome:
        """Parse a supported model-comparison question with disclosed defaults.

        Semantic identity remains fail-closed: this path never invents the
        physical target or changes the requested research type.  It only fills
        experiment-design fields that are already frozen by the active runtime
        capability registry.
        """
        if problem_type != "model_comparison":
            raise DeterministicProblemParserError(
                "safe_defaults_only_supported_for_model_comparison"
            )
        return self._parse(question, defaults=defaults)

    def _parse(
        self,
        question: str,
        *,
        defaults: dict[str, Any] | None,
    ) -> DeterministicParseOutcome:
        question = question.strip()
        if not question:
            raise DeterministicProblemParserError("research_question_required")
        text = question.casefold()
        if not any(term in text for term in (
            "蒸汽体积流量", "主蒸汽体积流量", "steam volumetric flow",
        )):
            raise DeterministicProblemParserError("explicit_supported_target_required")
        models = [name for name, pattern in self._MODELS if re.search(pattern, text)]
        references = [name for name in models if name == "persistence"]
        candidates = [name for name in models if name != "persistence"]
        horizon = self._horizon(text)
        completion_notes: list[str] = []
        field_sources = {
            "original_question": "USER",
            "target_variable": "USER",
            "required_models": "USER",
            "reference_models": "USER",
            "required_horizon_steps": "USER",
            "metrics": "USER",
            "required_operations": "SYSTEM_DEFAULT",
            "protocol_constraints": "SYSTEM_DEFAULT",
        }
        if defaults is not None and not candidates:
            candidates = list(defaults.get("candidate_models") or [])
            field_sources["required_models"] = "CAPABILITY_REGISTRY"
            completion_notes.append("candidate_models_from_capability_registry")
        if defaults is not None and not references:
            reference = str(defaults.get("reference_model") or "").strip()
            references = [reference] if reference else []
            field_sources["reference_models"] = "CAPABILITY_REGISTRY"
            completion_notes.append("reference_model_from_capability_registry")
        if not candidates or not references:
            raise DeterministicProblemParserError(
                "explicit_candidate_and_reference_models_required"
            )
        if defaults is not None and horizon is None:
            default_horizon = defaults.get("prediction_horizon_steps")
            horizon = int(default_horizon) if default_horizon is not None else None
            field_sources["required_horizon_steps"] = "CAPABILITY_REGISTRY"
            completion_notes.append("prediction_horizon_from_capability_registry")
        if horizon is None:
            raise DeterministicProblemParserError("explicit_prediction_horizon_required")
        metrics = [metric for metric in ("MAE", "RMSE", "R2", "MBE") if metric.casefold() in text]
        if defaults is not None and not metrics:
            metrics = [str(item).upper() for item in defaults.get("metrics") or []]
            field_sources["metrics"] = "CAPABILITY_REGISTRY"
            completion_notes.append("metrics_from_capability_registry")
        if not metrics:
            raise DeterministicProblemParserError("explicit_supported_metric_required")
        operations = ["model_comparison", "reference_model_comparison"]
        protocol = []
        locked_declared = (
            "锁定测试" in text or "locked_test" in text or "locked-test" in text
        )
        chronological_declared = "时间顺序" in text or "chronological" in text
        if locked_declared or defaults is not None:
            operations.append("locked_test_evaluation")
            protocol.append("locked_test_not_used_for_selection")
        if chronological_declared or defaults is not None:
            operations.append("chronological_validation")
            protocol.append("validation_only_model_selection")
        if not locked_declared and defaults is not None:
            completion_notes.append("locked_test_protocol_from_system_default")
        if not chronological_declared and defaults is not None:
            completion_notes.append("chronological_validation_from_system_default")
        if locked_declared and chronological_declared:
            field_sources["required_operations"] = "USER"
            field_sources["protocol_constraints"] = "USER"
        problem = ResearchProblemSpec(
            problem_id=f"P-{uuid4().hex[:12]}",
            original_question=question,
            research_object="锅炉蒸汽软测量",
            target_variable="steam_volumetric_flow",
            target_inference_reason="explicit_deterministic_target_match",
            objective="compare_prediction_accuracy",
            metrics=metrics,
            operating_condition=(
                "深度调峰" if "深度调峰" in question else "未特别限定"
            ),
            research_goal=question,
            success_criteria=["按预声明指标比较候选模型与参考模型"],
            required_models=candidates,
            reference_models=references,
            required_horizon_steps=horizon,
            required_operations=operations,
            protocol_constraints=protocol,
            required_objective_dimensions=["accuracy"],
        )
        return DeterministicParseOutcome(
            problem=problem,
            field_sources=field_sources,
            completion_notes=completion_notes,
        )
