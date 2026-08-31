from __future__ import annotations

from typing import Any

from boilermind.hypothesis.deterministic_admission import evaluate_candidate
from boilermind.ranking.historical_prior import score_hypothesis
from .base import BaseSkill


class RankingSkill(BaseSkill):
    """科学价值排序与当前执行能力映射相互独立。"""

    name = "hypothesis_ranking"

    @staticmethod
    def _score(hypothesis: dict[str, Any], problem: dict[str, Any]) -> dict[str, float]:
        canonical = score_hypothesis(hypothesis)
        return {
            "historical_support": canonical.historical_support,
            "historical_scope_match": canonical.historical_scope_match,
            "problem_relevance": canonical.problem_relevance,
            "reproducibility": canonical.reproducibility,
            "falsifiability": canonical.falsifiability,
            "prior_score": canonical.prior_score,
            "scientific_priority": round(canonical.prior_score * 100.0, 3),
        }

    @staticmethod
    def _verification_mapping(
        hypothesis: dict[str, Any],
        problem: dict[str, Any],
        admission: dict[str, Any],
    ) -> dict[str, Any]:
        history = hypothesis.get("historical_assessment", {})
        required_inputs: list[str] = []
        text = " ".join(str(hypothesis.get(key, "")) for key in (
            "hypothesis_statement", "expected_observation", "verification_intent",
        )).casefold()
        if any(term in text for term in ("预测", "估计", "mae", "rmse")):
            frozen_horizon = admission.get("required_horizon_steps")
            frozen_metrics = list(admission.get("required_metrics") or [])
            if frozen_horizon is None and problem.get("required_horizon_steps") is None:
                required_inputs.append("prediction_horizon")
            if not frozen_metrics and not problem.get("metrics"):
                required_inputs.append("evaluation_metric")
        duplicate = history.get("duplicate_status") == "DUPLICATE"
        conflicts = bool(history.get("conflicting_observations"))
        executable = not admission["missing_capabilities"] and not required_inputs and not duplicate
        if duplicate:
            action = "ALREADY_SUPPORTED"
        elif admission["missing_capabilities"]:
            action = "NEEDS_NEW_OPERATION"
        elif required_inputs:
            action = "ASK_USER"
        elif conflicts:
            action = "CONFLICT_REPLICATION"
        else:
            action = "EXECUTE_NOW"
        supported = sorted(
            set(admission["required_operations"])
            - {item.split(":", 1)[1] for item in admission["missing_capabilities"] if item.startswith("operation:")}
        )
        return {
            "supported_operations": supported,
            "missing_operations": [
                item.split(":", 1)[1] for item in admission["missing_capabilities"]
                if item.startswith("operation:")
            ],
            "missing_capabilities": list(admission["missing_capabilities"]),
            "required_user_inputs": required_inputs,
            "executable_now": executable,
            "recommended_action": action,
        }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        hypotheses = context.get("qualified_hypotheses") or context.get("hypotheses") or []
        if not hypotheses:
            raise RuntimeError("no_qualified_hypotheses_for_ranking")
        problem = context.get("research_problem") or {}
        scientific_context = context.get("scientific_context") or {}
        ranking: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            hid = str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or "").strip()
            if not hid:
                continue
            admission = evaluate_candidate(hypothesis, problem, scientific_context)
            scores = self._score(hypothesis, problem)
            mapping = self._verification_mapping(hypothesis, problem, admission)
            hypothesis["scientific_rank"] = dict(scores)
            hypothesis["verification_mapping"] = mapping
            hypothesis["workflow_status"] = "RANKED"
            ranking.append({
                "id": hid, "hypothesis_id": hid, **scores, **admission,
                "verification_mapping": mapping,
                "current_executable": bool(mapping["executable_now"]),
                "score": scores["scientific_priority"],
                "reason": (
                    "scientific_value_rank_independent_of_capability;"
                    f"action={mapping['recommended_action']}"
                ),
            })
        ranking.sort(key=lambda item: (
            -int(bool(item["current_executable"])),
            -item["scientific_priority"],
            item["hypothesis_id"],
        ))
        for index, item in enumerate(ranking, start=1):
            item["rank"] = index
        by_id = {item.get("hypothesis_id"): item for item in hypotheses}
        for item in ranking:
            if item["hypothesis_id"] in by_id:
                by_id[item["hypothesis_id"]]["scientific_rank"]["rank"] = item["rank"]
        champions = [
            item for item in ranking
            if item["verification_mapping"]["recommended_action"]
            in {"EXECUTE_NOW", "CONFLICT_REPLICATION"}
        ]
        selected = champions[0]["hypothesis_id"] if champions else None
        blockers = [] if selected else list(dict.fromkeys(
            item
            for ranked in ranking
            for item in (
                [ranked["verification_mapping"]["recommended_action"]]
                + ranked["verification_mapping"].get("missing_capabilities", [])
                + [
                    f"user_input:{value}"
                    for value in ranked["verification_mapping"].get(
                        "required_user_inputs", []
                    )
                ]
            )
        ))
        return {
            "ranking": ranking,
            "hypotheses": hypotheses,
            "qualified_hypotheses": hypotheses,
            "selected_hypothesis_id": selected,
            "ranking_type": "scientific_priority_then_capability_v1",
            "status": "ranked" if selected else "blocked_no_comprehensive_candidate",
            "selection_blockers": blockers,
        }
