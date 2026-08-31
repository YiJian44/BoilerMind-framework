"""Adapt a FullPipeline reporting context into generator input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from boilermind.core.contracts.base import ContractModel
from boilermind.core.contracts.evidence import EvidenceBundle
from boilermind.core.contracts.experiment import (
    ExperimentAudit,
    ExperimentContract,
    ExperimentPlan,
    ExperimentResult,
    ScientificResult,
)
from boilermind.core.contracts.hypothesis import ScientificHypothesis
from boilermind.core.contracts.research_problem import ResearchProblemSpec
from boilermind.core.contracts.scientific_research_plan import ResearchTraceEntry

from .scientific_research_plan_generator import (
    ScientificResearchPlanGeneratorInput,
)


class PipelineReportAdapterError(ValueError):
    """Raised when a pipeline context cannot safely produce a report input."""


ModelT = TypeVar("ModelT", bound=ContractModel)
_MISSING = object()


def _read(
    context: Mapping[str, Any],
    label: str,
    *aliases: str,
) -> Any:
    for name in aliases:
        value = context.get(name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    raise PipelineReportAdapterError(f"missing_required_object:{label}")


def _coerce(model: type[ModelT], value: Any, label: str) -> ModelT:
    if isinstance(value, model):
        return value
    try:
        return model.model_validate(value)
    except Exception as exc:
        raise PipelineReportAdapterError(
            f"invalid_required_object:{label}:{exc}"
        ) from exc


def _text(payload: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _strings(payload: Mapping[str, Any], *names: str) -> list[str]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return items
        if value is not None and str(value).strip():
            return [str(value).strip()]
    return []


def _coerce_reporting_hypothesis(
    value: Any,
    *,
    problem: ResearchProblemSpec,
    evidence: EvidenceBundle | None,
) -> ScientificHypothesis:
    """Normalize production hypotheses into the strict reporting contract."""

    if isinstance(value, ScientificHypothesis):
        return value
    if not isinstance(value, Mapping):
        return _coerce(ScientificHypothesis, value, "scientific_hypothesis")
    try:
        return ScientificHypothesis.model_validate(value)
    except Exception:
        pass

    hypothesis_id = _text(value, "hypothesis_id", "id")
    claim = _text(value, "hypothesis", "hypothesis_statement")
    mechanism = _text(
        value, "mechanism_chain", "engineering_mechanism", "mechanism"
    )
    evidence_ids = _strings(value, "evidence_ids")
    verified_ids = {
        item.evidence_id
        for item in (evidence.evidence if evidence else [])
        if item.citation_verified and item.semantic_verified
    }
    evidence_ids = [item for item in evidence_ids if item in verified_ids]
    observation_ids = _strings(value, "source_observation_ids")
    support_type = (
        "verified_evidence"
        if evidence_ids
        else "data_observation"
        if observation_ids
        else "hypothesis_inference"
    )
    model_family = _text(value, "model_family")
    normalized = {
        "hypothesis_id": hypothesis_id,
        "problem_id": _text(value, "problem_id") or problem.problem_id,
        "title": _text(value, "title") or claim,
        "research_significance": _text(value, "research_significance")
        or problem.research_goal
        or problem.objective,
        "hypothesis": claim,
        "mechanism_chain": mechanism,
        "mechanism_steps": [
            {
                "step": 1,
                "statement": mechanism,
                "support_type": support_type,
                "evidence_ids": evidence_ids,
            }
        ],
        "related_variables": _strings(
            value, "related_variables", "key_variables", "variables"
        ),
        "applicability_conditions": _strings(value, "applicability_conditions"),
        "verification_intent": _text(value, "verification_intent"),
        "expected_observation": _text(value, "expected_observation", "inference"),
        "confirmation_criteria": _strings(value, "confirmation_criteria"),
        "falsification_criteria": _strings(
            value, "falsification_criteria", "falsification_condition"
        ),
        "evidence_gaps": _strings(
            value, "evidence_gaps", "evidence_gap", "evidence_needed"
        ),
        "assumptions": _strings(value, "assumptions"),
        "counter_mechanisms": _strings(value, "counter_mechanisms"),
        "novelty_axis": _text(value, "novelty_axis")
        or (
            f"{model_family} 对 {problem.target_variable} 的数据属性适配性"
            if model_family
            else f"{problem.target_variable} 的可验证机理"
        ),
        "evidence_bundle_sha256": _text(value, "evidence_bundle_sha256")
        or (evidence.sha256 if evidence else "0" * 64),
    }
    try:
        return ScientificHypothesis.model_validate(normalized)
    except Exception as exc:
        raise PipelineReportAdapterError(
            f"invalid_required_object:scientific_hypothesis:{exc}"
        ) from exc


def resolve_selected_hypothesis(
    hypotheses: list[ScientificHypothesis],
    selected_hypothesis_id: str,
) -> ScientificHypothesis:
    """Resolve exactly one hypothesis by ID; never match on text."""

    if not selected_hypothesis_id or not selected_hypothesis_id.strip():
        raise PipelineReportAdapterError(
            "missing_required_object:selected_hypothesis_id"
        )
    matches = [
        item
        for item in hypotheses
        if item.hypothesis_id == selected_hypothesis_id
    ]
    if not matches:
        raise PipelineReportAdapterError(
            "selected_hypothesis_not_found:"
            f"{selected_hypothesis_id}"
        )
    if len(matches) != 1:
        raise PipelineReportAdapterError(
            "selected_hypothesis_not_unique:"
            f"{selected_hypothesis_id}"
        )
    return matches[0]


def _flatten_research_trace(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise PipelineReportAdapterError("invalid_required_object:research_trace")

    flattened: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping) and "research_trace" in item:
            nested = item["research_trace"]
            if not isinstance(nested, Sequence) or isinstance(
                nested, (str, bytes)
            ):
                raise PipelineReportAdapterError(
                    "invalid_required_object:research_trace"
                )
            candidates = nested
        else:
            candidates = [item]

        for candidate in candidates:
            try:
                entry = ResearchTraceEntry.model_validate(candidate)
            except Exception as exc:
                raise PipelineReportAdapterError(
                    f"invalid_required_object:research_trace:{exc}"
                ) from exc
            flattened.append(entry.model_dump(mode="python"))
    return flattened


def _same_id(label: str, *values: Any) -> None:
    normalized = {
        str(value)
        for value in values
        if value is not None and str(value).strip()
    }
    if len(normalized) != 1:
        raise PipelineReportAdapterError(f"id_mismatch:{label}")


class PipelineReportAdapter:
    """Convert existing FullPipeline facts without scientific inference."""

    def adapt(
        self,
        context: Mapping[str, Any],
    ) -> ScientificResearchPlanGeneratorInput:
        if not isinstance(context, Mapping):
            raise PipelineReportAdapterError("pipeline_context_must_be_mapping")

        problem = _coerce(
            ResearchProblemSpec,
            _read(context, "research_problem", "research_problem"),
            "research_problem",
        )
        raw_evidence = context.get("evidence_bundle")
        evidence = (
            _coerce(EvidenceBundle, raw_evidence, "evidence_bundle")
            if raw_evidence
            else None
        )
        raw_hypotheses = context.get(
            "qualified_hypotheses",
            context.get("hypotheses", _MISSING),
        )
        if raw_hypotheses is not _MISSING:
            if not isinstance(raw_hypotheses, Sequence) or isinstance(
                raw_hypotheses, (str, bytes)
            ):
                raise PipelineReportAdapterError(
                    "invalid_required_object:hypotheses"
                )
            hypotheses = [
                _coerce_reporting_hypothesis(
                    item,
                    problem=problem,
                    evidence=evidence,
                )
                for item in raw_hypotheses
            ]
            hypothesis = resolve_selected_hypothesis(
                hypotheses,
                str(
                    _read(
                        context,
                        "selected_hypothesis_id",
                        "selected_hypothesis_id",
                    )
                ),
            )
        else:
            hypothesis = _coerce_reporting_hypothesis(
                _read(
                    context,
                    "scientific_hypothesis",
                    "scientific_hypothesis",
                    "selected_hypothesis",
                    "hypothesis",
                ),
                problem=problem,
                evidence=evidence,
            )
        plan = _coerce(
            ExperimentPlan,
            _read(context, "experiment_plan", "experiment_plan", "plan"),
            "experiment_plan",
        )
        contract = _coerce(
            ExperimentContract,
            _read(
                context,
                "experiment_contract",
                "experiment_contract",
                "contract",
            ),
            "experiment_contract",
        )
        result = _coerce(
            ExperimentResult,
            _read(context, "experiment_result", "experiment_result"),
            "experiment_result",
        )
        audit = _coerce(
            ExperimentAudit,
            _read(
                context,
                "experiment_audit",
                "experiment_audit",
                "audit",
            ),
            "experiment_audit",
        )
        scientific = _coerce(
            ScientificResult,
            _read(context, "scientific_result", "scientific_result"),
            "scientific_result",
        )
        trace = _flatten_research_trace(
            _read(context, "research_trace", "research_trace")
        )

        _same_id(
            "problem_id",
            problem.problem_id,
            evidence.problem_id if evidence else None,
            hypothesis.problem_id,
            plan.problem_id,
            contract.problem_id,
            result.problem_id,
        )
        _same_id(
            "hypothesis_id",
            hypothesis.hypothesis_id,
            context.get("selected_hypothesis_id"),
            plan.hypothesis_id,
            contract.hypothesis_id,
            result.hypothesis_id,
            scientific.hypothesis_id,
        )
        _same_id("plan_id", plan.plan_id, contract.plan_id, result.plan_id)
        _same_id(
            "experiment_id",
            contract.experiment_id,
            result.experiment_id,
            audit.experiment_id,
            scientific.experiment_id,
        )
        if (
            evidence is not None
            and hypothesis.evidence_bundle_sha256
            and hypothesis.evidence_bundle_sha256 != evidence.sha256
        ):
            raise PipelineReportAdapterError(
                "evidence_bundle_sha256_mismatch"
            )

        validity_source = _read(
            context,
            "validity_source",
            "validity_source",
        )
        if validity_source != "ExperimentAudit":
            raise PipelineReportAdapterError(
                "invalid_validity_source:ExperimentAudit_required"
            )
        audit_valid = _read(
            context,
            "experiment_valid",
            "experiment_valid",
        )
        if not isinstance(audit_valid, bool):
            raise PipelineReportAdapterError(
                "invalid_required_object:experiment_valid"
            )
        if result.experiment_valid is None:
            validity_issues = list(result.experiment_validity_issues)
            if not audit_valid and not validity_issues:
                validity_issues = list(audit.issues)
            result = result.model_copy(
                update={
                    "experiment_valid": audit_valid,
                    "experiment_validity_issues": validity_issues,
                }
            )
        elif result.experiment_valid is not audit_valid:
            raise PipelineReportAdapterError(
                "experiment_valid_audit_mismatch"
            )

        return ScientificResearchPlanGeneratorInput(
            research_problem=problem,
            evidence_bundle=evidence,
            hypothesis=hypothesis,
            experiment_plan=plan,
            experiment_contract=contract,
            experiment_result=result,
            experiment_audit=audit,
            scientific_result=scientific,
            research_trace=trace,
        )


def adapt_full_pipeline_report_context(
    context: Mapping[str, Any],
) -> ScientificResearchPlanGeneratorInput:
    """Functional entry point for the strict pipeline adapter."""

    return PipelineReportAdapter().adapt(context)
