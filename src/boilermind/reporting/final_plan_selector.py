"""Deterministically select the one final hypothesis/plan/reporting round."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from boilermind.core.contracts.base import ContractModel
from boilermind.core.contracts.scientific_research_plan import FinalPlanSelection


class FinalPlanSelectionResult(ContractModel):
    selection: FinalPlanSelection
    hypothesis: dict[str, Any]
    member: dict[str, Any]


class FinalResearchPlanSelector:
    """Prefer supported outcomes, otherwise the latest valid ranked outcome."""

    def select(self, state: Mapping[str, Any]) -> FinalPlanSelectionResult:
        hypotheses = {
            str(item.get("hypothesis_id")): dict(item)
            for item in state.get("hypotheses", [])
            if isinstance(item, Mapping) and item.get("hypothesis_id")
        }
        scores = {}
        snapshots = state.get("ranking_snapshots", [])
        if snapshots:
            latest = snapshots[-1]
            entries = latest.get("entries", []) if isinstance(latest, Mapping) else getattr(latest, "entries", [])
            for entry in entries:
                payload = entry if isinstance(entry, Mapping) else entry.model_dump(mode="json")
                scores[str(payload.get("hypothesis_id"))] = float(payload.get("dynamic_score", 0.0))

        valid: list[tuple[int, int, float, dict[str, Any], str]] = []
        observed_rounds: list[int] = []
        for batch in state.get("batches", []):
            batch_payload = batch if isinstance(batch, Mapping) else batch.model_dump(mode="json")
            round_index = int(batch_payload.get("round_index", 1))
            observed_rounds.append(round_index)
            for member in batch_payload.get("members", []):
                payload = member if isinstance(member, Mapping) else member.model_dump(mode="json")
                outcome = payload.get("outcome") or {}
                audit = outcome.get("audit") or {}
                result = outcome.get("experiment_result") or {}
                if payload.get("status") != "COMPLETED" or not audit.get("execution_valid"):
                    continue
                if str(result.get("status", "")).lower() == "failed":
                    continue
                hypothesis_id = str(payload.get("hypothesis_id", ""))
                verdict = str((outcome.get("scientific_result") or {}).get("verdict", "")).upper()
                supported = 1 if verdict == "SUPPORTED" else 0
                valid.append((supported, round_index, scores.get(hypothesis_id, 0.0), dict(payload), hypothesis_id))
        if not valid:
            raise ValueError("no_valid_completed_experiment_for_scientific_plan")
        supported = [item for item in valid if item[0] == 1]
        pool = supported or valid
        chosen = max(pool, key=lambda item: (item[1], item[2], item[4]))
        _, round_index, _, member, hypothesis_id = chosen
        attempted_iteration = bool(observed_rounds and max(observed_rounds) > 1)
        fallback = bool(observed_rounds and max(observed_rounds) > round_index)
        hypothesis = hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"selected_hypothesis_not_found:{hypothesis_id}")
        contract = member.get("contract") or {}
        plan = member.get("plan") or {}
        return FinalPlanSelectionResult(
            selection=FinalPlanSelection(
                hypothesis_id=hypothesis_id,
                round_index=round_index,
                revision_index=max(0, round_index - 1),
                plan_id=str(plan.get("plan_id") or contract.get("plan_id")),
                experiment_id=contract.get("experiment_id"),
                selection_reason=("FALLBACK_TO_LAST_VALID_ROUND" if fallback else
                                  "LATEST_VALID_REVISION" if round_index > 1 else
                                  "FIRST_ROUND_NO_ITERATION"),
                iteration_occurred=attempted_iteration,
                fallback_applied=fallback,
            ),
            hypothesis=hypothesis,
            member=member,
        )
