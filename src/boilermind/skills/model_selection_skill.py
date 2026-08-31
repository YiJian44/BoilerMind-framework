from __future__ import annotations

from typing import Any

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)
from boilermind.models import build_default_registry

from .base import BaseSkill


_COMPUTE_COST_RANK = {
    "trivial": 0,
    "seconds": 1,
    "minutes": 2,
    "hours": 3,
}


class ModelSelectionSkill(BaseSkill):
    """Deterministically select only registered soft-sensing models."""

    name = "model_selection"
    description = "从 ModelRegistry 选择候选模型与基线模型"

    @staticmethod
    def infer_task_type(target_variable: str, objective: str) -> str:
        text = f"{target_variable} {objective}".casefold()
        if any(token in text for token in ("优化", "optimization", "给煤", "送风")):
            return "optimization"
        if any(token in text for token in ("故障", "诊断", "diagnos", "fault")):
            return "diagnosis"
        return "prediction"

    def __init__(
        self,
        *,
        model_registry: Any | None = None,
        capability_registry: ExperimentCapabilityRegistry | None = None,
        max_candidates: int = 3,
    ):
        self.model_registry = model_registry or build_default_registry()
        if capability_registry is not None:
            self.capability_registry = capability_registry
        else:
            try:
                self.capability_registry = ExperimentCapabilityRegistry()
            except FileNotFoundError:
                # Catalog inspection must remain possible without local data,
                # but executability then fails closed to callable adapters.
                self.capability_registry = None
        self.max_candidates = max_candidates

    @staticmethod
    def _ordered(specs):
        return sorted(
            specs,
            key=lambda spec: (
                _COMPUTE_COST_RANK.get(spec.compute_cost, 9),
                spec.model_name,
            ),
        )

    @staticmethod
    def _required_capability_tags(
        hypothesis_statement: str,
        mechanism_chain: str | list[Any],
        verification_intent: str,
        objective: str,
        task_type: str,
    ) -> set[str]:
        mechanism = (
            " ".join(str(item) for item in mechanism_chain)
            if isinstance(mechanism_chain, list)
            else str(mechanism_chain)
        )
        text = " ".join((
            str(hypothesis_statement), mechanism,
            str(verification_intent), str(objective),
        )).casefold()
        tags: set[str] = set()
        if any(token in text for token in (
            "历史", "窗口", "滞后", "时序", "temporal", "sequence", "lag",
        )):
            tags.update({"temporal_dependency", "sequence_model"})
        if any(token in text for token in (
            "物理", "守恒", "机理", "if97", "physics",
        )):
            tags.add("physics_constraint")
        if any(token in text for token in (
            "特征选择", "变量筛选", "feature selection", "sparse",
        )):
            tags.add("feature_selection")
        if any(token in text for token in (
            "不确定", "置信区间", "uncertainty", "probabilistic",
        )):
            tags.add("uncertainty_estimation")
        if task_type == "optimization":
            tags.add("optimization_surrogate")
        return tags

    def _rank_candidates(self, specs, required_tags: set[str]):
        return sorted(
            specs,
            key=lambda spec: (
                -len(required_tags & set(spec.capability_tags)),
                _COMPUTE_COST_RANK.get(spec.compute_cost, 9),
                spec.model_name,
            ),
        )

    def _select_top_candidates(self, specs, required_tags: set[str]):
        ranked = self._rank_candidates(specs, required_tags)
        chosen = []
        families = set()
        for spec in ranked:
            family = spec.family or spec.framework
            if family in families:
                continue
            chosen.append(spec)
            families.add(family)
            if len(chosen) >= self.max_candidates:
                return chosen
        for spec in ranked:
            if spec not in chosen:
                chosen.append(spec)
            if len(chosen) >= self.max_candidates:
                break
        return chosen

    def _choose_from_pool(
        self,
        pool,
        required_tags: set[str],
        required_models: set[str],
    ) -> tuple[list[str], list[str]]:
        persistence = [spec for spec in pool if spec.model_name == "persistence"]
        if persistence:
            baselines = ["persistence"]
            candidate_pool = [spec for spec in pool if spec.model_name != "persistence"]
        else:
            ordered = self._ordered(pool)
            ridge = [spec for spec in ordered if spec.model_name == "ridge"]
            baseline = ridge[0] if ridge else (ordered[0] if ordered else None)
            baselines = [baseline.model_name] if baseline else []
            candidate_pool = [spec for spec in ordered if spec is not baseline]

        required_candidates = required_models - set(baselines)
        if required_candidates:
            available = {spec.model_name for spec in candidate_pool}
            candidates = sorted(required_candidates & available)
        else:
            candidates = [
                spec.model_name
                for spec in self._select_top_candidates(candidate_pool, required_tags)
            ]
        return candidates, baselines

    def select_models(
        self,
        *,
        hypothesis_statement: str,
        mechanism_chain: str | list[Any],
        verification_intent: str,
        task_type: str,
        target_variable: str,
        operating_condition: str = "",
        objective: str = "",
        metrics: list[str] | None = None,
        required_models: list[str] | None = None,
    ) -> dict[str, Any]:
        task = str(task_type).strip().casefold()
        if str(target_variable).strip().casefold() in {
            "", "unspecified", "unknown", "none", "null",
        }:
            raise ValueError("target_variable_resolution_required")
        target = self.model_registry.resolve_target(target_variable)
        if target is None:
            return {
                "candidate_models": [],
                "baseline_models": [],
                "selection_reason": f"unsupported_target_variable:{target_variable}",
                "missing_models": [],
                "canonical_target": None,
                "task_type": task,
            }

        metric_list = [str(item) for item in (metrics or ["MAE", "RMSE"])]
        required_model_set = {
            str(item).strip().lower()
            for item in (required_models or [])
            if str(item).strip()
        }

        theoretical_pool = self.model_registry.match_task_capability(
            task_type=task,
            target_variable=target,
            metrics=metric_list,
            capability=None,
        )

        pool_names = {spec.model_name.lower() for spec in theoretical_pool}
        missing_models = sorted(required_model_set - pool_names)
        if missing_models:
            return {
                "candidate_models": [],
                "baseline_models": [],
                "selection_reason": "required_models_not_available",
                "missing_models": missing_models,
                "canonical_target": target,
                "task_type": task,
            }

        required_tags = self._required_capability_tags(
            hypothesis_statement,
            mechanism_chain,
            verification_intent,
            objective,
            task,
        )

        recommended_models, theoretical_baselines = self._choose_from_pool(
            theoretical_pool,
            required_tags,
            required_model_set,
        )
        executable_pool = self.model_registry.match_task_capability(
            task_type=task,
            target_variable=target,
            metrics=metric_list,
            capability=self.capability_registry,
        )
        if self.capability_registry is None:
            executable_pool = [
                spec for spec in executable_pool
                if spec.runner_callable and spec.adapter_available
            ]
        executable_models, baseline_models = self._choose_from_pool(
            executable_pool,
            required_tags,
            required_model_set,
        )

        if not theoretical_baselines or not recommended_models:
            return {
                "candidate_models": [],
                "baseline_models": [],
                "recommended_models": [],
                "executable_models": [],
                "model_substitution_reason": "",
                "selection_reason": "registered_model_pool_insufficient",
                "missing_models": [],
                "canonical_target": target,
                "task_type": task,
            }

        selected = (
            set(recommended_models)
            | set(executable_models)
            | set(theoretical_baselines)
            | set(baseline_models)
        )
        if not selected.issubset(set(self.model_registry.names())):
            raise RuntimeError("model_selection_contains_unregistered_model")

        candidate_reasons = []
        model_scores = {}
        for name in recommended_models:
            spec = self.model_registry.get(name)
            matched = sorted(required_tags & set(spec.capability_tags))
            model_scores[name] = len(matched)
            candidate_reasons.append(
                f"{name}: matched capabilities="
                f"{','.join(matched) if matched else 'general_task_target_fit'}"
                f"; family={spec.family or spec.framework}"
            )

        rationale = (
            "Theoretical Top-N models selected from ModelRegistry by "
            "hypothesis-mechanism "
            f"fit for task={task}, target={target}. Required capabilities="
            f"{','.join(sorted(required_tags)) if required_tags else 'general_task_fit'}. "
            + " ".join(candidate_reasons)
            + f" Scientific baseline={','.join(theoretical_baselines)} provides a simpler "
            "registered reference for the same task/target. "
            f"Operating condition={str(operating_condition).strip() or 'unspecified'}; "
            f"verification intent={str(verification_intent).strip() or 'unspecified'}."
        )
        if task == "optimization":
            rationale += " Selected models are optimization surrogates, not control solvers."

        if recommended_models == executable_models:
            substitution_reason = (
                "No substitution: all scientifically recommended models are "
                "currently executable."
            )
        elif executable_models:
            unavailable = [
                name for name in recommended_models
                if name not in executable_models
            ]
            substitution_reason = (
                "Current CapabilityRegistry cannot execute recommended models="
                f"{','.join(unavailable)} under the frozen runtime/data contract; "
                f"execution uses={','.join(executable_models)}. The original "
                "recommended_models are preserved and not overwritten."
            )
        else:
            substitution_reason = (
                "No currently executable models satisfy the frozen capability "
                "contract; recommended_models are preserved for scientific review."
            )

        return {
            "candidate_models": executable_models,
            "baseline_models": baseline_models,
            "recommended_models": recommended_models,
            "executable_models": executable_models,
            "recommended_baseline_models": theoretical_baselines,
            "model_substitution_reason": substitution_reason,
            "selection_reason": rationale,
            "missing_models": [],
            "canonical_target": target,
            "task_type": task,
            "required_capability_tags": sorted(required_tags),
            "model_scores": model_scores,
        }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        hypothesis = context.get("hypothesis", {})
        if hasattr(hypothesis, "model_dump"):
            hypothesis = hypothesis.model_dump(mode="json")
        if not isinstance(hypothesis, dict):
            hypothesis = {}
        return self.select_models(
            hypothesis_statement=str(
                context.get("hypothesis_statement")
                or hypothesis.get("hypothesis")
                or hypothesis.get("statement")
                or ""
            ),
            mechanism_chain=(
                context.get("mechanism_chain")
                or hypothesis.get("mechanism_chain")
                or hypothesis.get("mechanism_steps")
                or ""
            ),
            verification_intent=str(
                context.get("verification_intent")
                or hypothesis.get("verification_intent")
                or ""
            ),
            task_type=str(context.get("task_type", "prediction")),
            target_variable=str(context.get("target_variable", "")),
            operating_condition=str(context.get("operating_condition", "")),
            objective=str(context.get("objective", "")),
            metrics=context.get("metrics"),
            required_models=context.get("required_models"),
        )
