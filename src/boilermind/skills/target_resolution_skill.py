from __future__ import annotations

from typing import Any

from .base import BaseSkill


class TargetResolutionSkill(BaseSkill):
    """Resolve a prediction target from supplied scientific/runtime context."""

    name = "target_resolution"
    description = "在模型选择前解析实际预测目标；不确定时关闭执行链路"

    _UNRESOLVED = {"", "unspecified", "unknown", "none", "null"}
    _PREDICTION_TERMS = (
        "预测", "软测量", "forecast", "prediction", "predict",
    )
    _ACCURACY_TERMS = (
        "预测精度", "准确率提升", "提高准确率", "误差降低", "降低误差",
        "prediction accuracy", "accuracy improvement", "lower error",
    )
    _OPTIMIZATION_TERMS = (
        "优化", "最优", "optimization", "optimize", "minimum", "maximum",
    )

    @staticmethod
    def infer_task_type(*values: Any) -> str:
        text = " ".join(str(value) for value in values).casefold()
        if any(token in text for token in ("优化", "optimization", "给煤", "送风")):
            return "optimization"
        if any(token in text for token in ("故障", "诊断", "diagnos", "fault")):
            return "diagnosis"
        return "prediction"

    @staticmethod
    def _strings(value: Any) -> list[str]:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            result: list[str] = []
            for item in value.values():
                result.extend(TargetResolutionSkill._strings(item))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(TargetResolutionSkill._strings(item))
            return result
        return []

    @staticmethod
    def _schema_targets(schema: Any) -> list[str]:
        if not isinstance(schema, dict):
            return []
        values: list[str] = []
        for key in ("target_variables", "target_columns", "targets"):
            raw = schema.get(key, [])
            raw = [raw] if isinstance(raw, str) else raw
            if isinstance(raw, list):
                values.extend(str(item).strip() for item in raw if str(item).strip())
        columns = schema.get("columns", [])
        if isinstance(columns, list):
            for column in columns:
                if isinstance(column, dict) and str(column.get("role", "")).casefold() in {
                    "target", "label", "prediction_target",
                }:
                    name = str(column.get("name", "")).strip()
                    if name:
                        values.append(name)
        return list(dict.fromkeys(values))

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        problem = context.get("research_problem") or {}
        if hasattr(problem, "model_dump"):
            problem = problem.model_dump(mode="json")
        if not isinstance(problem, dict):
            raise ValueError("research_problem_required")

        current = str(problem.get("target_variable", "")).strip()
        objective = str(problem.get("objective", "") or "unspecified").strip()
        metrics = [str(item) for item in (problem.get("metrics") or [])]
        if current.casefold() not in self._UNRESOLVED:
            return {
                "target_variable": current,
                "objective": objective,
                "metrics": metrics,
                "target_inference_reason": "target_explicitly_structured_by_problem_parser",
                "resolved": True,
                "status": "target_resolved",
            }

        question = str(problem.get("original_question", ""))
        research_object = str(problem.get("research_object", ""))
        task_type = str(context.get("task_type", "")).casefold()
        scientific_text = " ".join(
            [question, research_object]
            + self._strings(context.get("evidence_bundle"))
            + self._strings(context.get("historical_research_context"))
        ).casefold()

        schema_targets = self._schema_targets(context.get("current_data_schema"))
        history_targets = []
        historical_context = context.get("historical_research_context") or []
        if isinstance(historical_context, dict):
            historical_context = [historical_context]
        for item in historical_context:
            if isinstance(item, dict):
                value = str(item.get("target_variable", "")).strip()
                if value and value.casefold() not in self._UNRESOLVED:
                    history_targets.append(value)
        candidates = list(dict.fromkeys(schema_targets + history_targets))

        prediction_task = (
            task_type == "prediction"
            or any(term in scientific_text for term in self._PREDICTION_TERMS)
        )
        optimization_task = (
            task_type == "optimization"
            or any(term in scientific_text for term in self._OPTIMIZATION_TERMS)
        )
        mentioned = [candidate for candidate in candidates if candidate.casefold() in scientific_text]

        resolved: str | None = None
        reason = ""
        if prediction_task and len(mentioned) == 1:
            resolved = mentioned[0]
            reason = "single_target_supported_by_question_evidence_or_history_and_schema"
        elif prediction_task and "软测量" in scientific_text and len(schema_targets) == 1:
            resolved = schema_targets[0]
            reason = "soft_sensor_prediction_with_single_declared_dataset_target"
        elif prediction_task and len(set(history_targets)) == 1 and history_targets[0] in candidates:
            resolved = history_targets[0]
            reason = "single_target_supported_by_historical_research_context"
        elif optimization_task and len(schema_targets) == 1:
            resolved = schema_targets[0]
            reason = "optimization_task_with_single_declared_schema_objective"

        if resolved is None:
            return {
                "target_variable": "unspecified",
                "objective": objective,
                "metrics": metrics,
                "target_inference_reason": (
                    "insufficient_or_ambiguous_target_evidence; candidates="
                    + ",".join(candidates)
                ),
                "resolved": False,
                "status": "target_variable_resolution_failed",
            }

        if any(term in question.casefold() for term in self._ACCURACY_TERMS):
            objective = "improve_prediction_accuracy"
            metrics = ["MAE", "RMSE", "R2"]
        elif optimization_task and not metrics:
            schema_metrics = context.get("current_data_schema", {}).get("metrics", [])
            if isinstance(schema_metrics, list):
                metrics = [str(item) for item in schema_metrics]

        return {
            "target_variable": resolved,
            "objective": objective,
            "metrics": metrics,
            "target_inference_reason": reason,
            "resolved": True,
            "status": "target_resolved",
        }
