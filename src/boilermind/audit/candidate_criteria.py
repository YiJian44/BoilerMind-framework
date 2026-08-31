from __future__ import annotations

import re
from typing import Any

from boilermind.audit.criterion_assessment import (
    CriterionAssessment,
)

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
)


_CONFIRMATION_PATTERN = re.compile(
    r"^all_candidates_worse_than_reference_on:"
    r"(.+)$"
)

_FALSIFICATION_PATTERN = re.compile(
    r"^any_candidate_better_than_reference_on:"
    r"(.+)$"
)

_RELATION_PATTERN = re.compile(
    r"^(all_models_not_better_than_model_on|"
    r"any_model_better_than_model_on|"
    r"all_models_better_than_model_on|"
    r"any_model_not_better_than_model_on):"
    r"([^|]+)\|([^|]+)\|(.+)$"
)

_REGIME_PATTERN = re.compile(
    r"^(all_models_regime_metric_greater|"
    r"all_models_regime_metric_not_greater):"
    r"([^|]+)\|([^|]+)\|(.+)$"
)


def _regime_criterion(
    criterion: str,
    result: ExperimentResult,
) -> tuple[bool, str] | None:
    match = _REGIME_PATTERN.match(criterion)
    if not match:
        return None
    relation, treatment, control, metric = match.groups()
    metric = metric.strip().upper()
    comparisons = {}
    for model, values in result.regime_metrics.items():
        try:
            treatment_value = float(values[treatment.strip()][metric])
            control_value = float(values[control.strip()][metric])
        except KeyError as exc:
            raise ValueError(f"regime_metric_missing:{model}:{exc}") from exc
        comparisons[model] = treatment_value > control_value
    if not comparisons:
        raise ValueError("regime_model_metrics_missing")
    if relation == "all_models_regime_metric_greater":
        met = all(comparisons.values())
    else:
        met = not any(comparisons.values())
    detail = ",".join(
        f"{model}:{'greater' if greater else 'not_greater'}"
        for model, greater in sorted(comparisons.items())
    )
    return met, f"{criterion}:{detail}"


def _metric_values(
    result: ExperimentResult,
    model_id: str,
    metrics: list[str],
) -> dict[str, float]:

    candidate_metrics = (
        result.candidate_locked_test_metrics
    )

    if model_id not in candidate_metrics:
        raise ValueError(
            "candidate_locked_test_metrics_missing:"
            f"{model_id}"
        )

    values = {}

    for metric in metrics:
        if metric not in candidate_metrics[model_id]:
            raise ValueError(
                "candidate_metric_missing:"
                f"{model_id}:{metric}"
            )

        values[metric] = float(
            candidate_metrics[model_id][metric]
        )

    return values


def _parse_confirmation(
    criterion: str,
) -> list[str]:
    match = _CONFIRMATION_PATTERN.match(
        criterion
    )

    if not match:
        raise ValueError(
            "unsupported_confirmation_criterion:"
            f"{criterion}"
        )

    return [
        metric.strip().upper()
        for metric in match.group(1).split(",")
        if metric.strip()
    ]


def _parse_falsification(
    criterion: str,
) -> list[str]:
    match = _FALSIFICATION_PATTERN.match(
        criterion
    )

    if not match:
        raise ValueError(
            "unsupported_falsification_criterion:"
            f"{criterion}"
        )

    return [
        metric.strip().upper()
        for metric in match.group(1).split(",")
        if metric.strip()
    ]


def _worse_on_all(
    candidate: dict[str, float],
    reference: dict[str, float],
) -> bool:
    return all(
        (
            candidate[metric] < reference[metric]
            if metric == "R2"
            else abs(candidate[metric]) > abs(reference[metric])
            if metric == "MBE"
            else candidate[metric] > reference[metric]
        )
        for metric in candidate
    )


def _better_on_all(
    candidate: dict[str, float],
    reference: dict[str, float],
) -> bool:
    return all(
        (
            candidate[metric] > reference[metric]
            if metric == "R2"
            else abs(candidate[metric]) < abs(reference[metric])
            if metric == "MBE"
            else candidate[metric] < reference[metric]
        )
        for metric in candidate
    )


def _relation_criterion(
    criterion: str,
    result: ExperimentResult,
) -> tuple[bool, str] | None:
    match = _RELATION_PATTERN.match(criterion)
    if not match:
        return None
    relation, model_csv, reference_model, metric_csv = match.groups()
    models = [item.strip() for item in model_csv.split(",") if item.strip()]
    metrics = [item.strip().upper() for item in metric_csv.split(",") if item.strip()]
    reference = _metric_values(result, reference_model.strip(), metrics)
    better = []
    not_better = []
    for model in models:
        values = _metric_values(result, model, metrics)
        if _better_on_all(values, reference):
            better.append(model)
        else:
            not_better.append(model)
    if relation == "all_models_not_better_than_model_on":
        met = not better
    elif relation == "any_model_better_than_model_on":
        met = bool(better)
    elif relation == "all_models_better_than_model_on":
        met = not not_better
    else:
        met = bool(not_better)
    reason = f"{criterion}:better={','.join(better)};not_better={','.join(not_better)}"
    return met, reason


def assess_candidate_locked_test_criteria(
    contract: ExperimentContract,
    result: ExperimentResult,
) -> CriterionAssessment:
    """
    Deterministic H002-style criterion assessment.

    Reads ONLY:
      - contract.confirmation_criteria /
        contract.falsification_criteria
      - result.candidate_locked_test_metrics

    - all candidates worse than reference on every listed
      metric  -> confirmation
    - any candidate better than reference on every listed
      metric  -> falsification
    - mixed    -> neither (INSUFFICIENT_EVIDENCE downstream)

    Unknown criteria format or missing metrics -> fail closed.
    """

    if (
        not contract.confirmation_criteria
        or not contract.falsification_criteria
    ):
        raise ValueError(
            "confirmation_and_falsification_criteria_required"
        )

    reference_models = (
        contract.reference_models
        or ["persistence"]
    )

    reference_model = reference_models[0]

    candidate_models = list(
        contract.candidate_models
    )

    if not candidate_models:
        raise ValueError(
            "candidate_models_required"
        )

    # Every predeclared confirmation criterion must hold.
    confirmation_met = True
    achieved_criteria: list[str] = []
    confirmation_reasons: list[str] = []

    for criterion in contract.confirmation_criteria:
        regime_relation = _regime_criterion(criterion, result)
        if regime_relation is not None:
            criterion_met, reason = regime_relation
            if criterion_met:
                achieved_criteria.append(criterion)
            else:
                confirmation_met = False
            confirmation_reasons.append(reason)
            continue
        relation = _relation_criterion(criterion, result)
        if relation is not None:
            criterion_met, reason = relation
            if criterion_met:
                achieved_criteria.append(criterion)
            else:
                confirmation_met = False
            confirmation_reasons.append(reason)
            continue
        metrics = _parse_confirmation(criterion)

        reference = _metric_values(
            result,
            reference_model,
            metrics,
        )

        candidates_worse = []
        candidates_not_worse = []

        for model_id in candidate_models:
            candidate = _metric_values(
                result,
                model_id,
                metrics,
            )

            if _worse_on_all(
                candidate,
                reference,
            ):
                candidates_worse.append(model_id)
            else:
                candidates_not_worse.append(model_id)

        criterion_met = (
            not candidates_not_worse
        )

        if criterion_met:
            achieved_criteria.append(
                criterion
            )
        else:
            confirmation_met = False

        confirmation_reasons.append(
            f"{criterion}:worse="
            f"{','.join(candidates_worse)};"
            f"not_worse="
            f"{','.join(candidates_not_worse)}"
        )

    # Any predeclared falsification criterion ends the
    # hypothesis in the falsified direction.
    falsification_met = False
    failed_criteria: list[str] = []
    falsification_reasons: list[str] = []

    for criterion in contract.falsification_criteria:
        regime_relation = _regime_criterion(criterion, result)
        if regime_relation is not None:
            criterion_met, reason = regime_relation
            if criterion_met:
                failed_criteria.append(criterion)
                falsification_met = True
            falsification_reasons.append(reason)
            continue
        relation = _relation_criterion(criterion, result)
        if relation is not None:
            criterion_met, reason = relation
            if criterion_met:
                falsification_met = True
                failed_criteria.append(criterion)
            falsification_reasons.append(reason)
            continue
        metrics = _parse_falsification(criterion)

        reference = _metric_values(
            result,
            reference_model,
            metrics,
        )

        better_candidates = []

        for model_id in candidate_models:
            candidate = _metric_values(
                result,
                model_id,
                metrics,
            )

            if _better_on_all(
                candidate,
                reference,
            ):
                better_candidates.append(model_id)

        criterion_met = bool(better_candidates)

        if criterion_met:
            falsification_met = True
            failed_criteria.append(criterion)

        falsification_reasons.append(
            f"{criterion}:better="
            f"{','.join(better_candidates)}"
        )

    rationale = (
        "candidate_locked_test_criteria:"
        + "|".join(confirmation_reasons)
        + ";"
        + "|".join(falsification_reasons)
    )

    return CriterionAssessment(
        experiment_id=result.experiment_id,
        confirmation_met=confirmation_met,
        falsification_met=falsification_met,
        achieved_criteria=achieved_criteria,
        failed_criteria=failed_criteria,
        rationale=rationale,
    )
