from __future__ import annotations

from typing import Any

from boilermind.core.contracts import (
    ExperimentContract,
    ExperimentResult,
    ScientificResult,
)
from boilermind.core.enums import ExperimentStatus

from .experiment_reflector import reflect_experiment
from .schemas import (
    ExperimentOptimizationSuggestion,
    FORECAST_HORIZONS,
    LEARNING_RATES,
    MODEL_WIDTHS,
    WINDOW_SIZES,
    WhitelistedConfiguration,
)


def _next_value(current: int | float, allowed: tuple[int | float, ...]):
    return next((value for value in allowed if value > current), current)


def _single_model(contract: ExperimentContract) -> str | None:
    return contract.candidate_models[0] if len(contract.candidate_models) == 1 else None


def _forecast_minutes(contract: ExperimentContract) -> int | None:
    steps = contract.prediction_horizon_steps
    interval = contract.sampling_interval_seconds
    if steps is None or interval is None:
        return None
    minutes = steps * interval / 60
    return int(minutes) if minutes.is_integer() and int(minutes) in FORECAST_HORIZONS else None


def _model_configuration(
    contract: ExperimentContract,
    result: ExperimentResult,
    model_name: str | None,
) -> dict[str, Any]:
    if model_name and model_name in result.model_records:
        return dict(result.model_records[model_name].model_configuration)
    requirements = contract.execution_requirements.get("model_configuration", {})
    return dict(requirements) if isinstance(requirements, dict) else {}


def _current_configuration(
    contract: ExperimentContract,
    result: ExperimentResult,
) -> dict[str, Any]:
    model_name = _single_model(contract)
    model_config = _model_configuration(contract, result, model_name)
    current: dict[str, Any] = {
        "model_name": model_name,
        "window_size": contract.window_steps,
        "forecast_horizon": _forecast_minutes(contract),
    }
    for name in ("d_model", "hidden_size", "learning_rate"):
        if name in model_config:
            current[name] = model_config[name]
    return current


def _safe_next_configuration(current: dict[str, Any], adjust: bool):
    model = str(current.get("model_name") or "").casefold() or None
    payload: dict[str, Any] = {"model_name": model}
    window = current.get("window_size")
    window_changed = False
    if window in WINDOW_SIZES:
        next_window = (
            _next_value(window, WINDOW_SIZES) if adjust else window
        )
        payload["window_size"] = next_window
        window_changed = next_window != window
    horizon = current.get("forecast_horizon")
    if horizon in FORECAST_HORIZONS:
        # Forecast horizon is task semantics and is preserved, not tuned.
        payload["forecast_horizon"] = horizon
    learning_rate = current.get("learning_rate")
    if learning_rate in LEARNING_RATES:
        payload["learning_rate"] = learning_rate
    if model == "transformer" and current.get("d_model") in MODEL_WIDTHS:
        width = current["d_model"]
        payload["d_model"] = (
            _next_value(width, MODEL_WIDTHS)
            if adjust and not window_changed else width
        )
    if model in {"lstm", "gru"} and current.get("hidden_size") in MODEL_WIDTHS:
        width = current["hidden_size"]
        payload["hidden_size"] = (
            _next_value(width, MODEL_WIDTHS)
            if adjust and not window_changed else width
        )
    return WhitelistedConfiguration.model_validate(payload)


def optimize_experiment_parameters(
    contract: ExperimentContract,
    result: ExperimentResult,
    scientific_result: ScientificResult,
) -> ExperimentOptimizationSuggestion:
    """Return a whitelist-only suggestion without changing any input object."""
    reflection = reflect_experiment(contract, result, scientific_result)
    current = _current_configuration(contract, result)
    next_configuration = _safe_next_configuration(current, reflection.should_adjust)
    next_payload = next_configuration.model_dump(exclude_none=True)
    changed = [
        name for name, value in next_payload.items()
        if name != "model_name" and current.get(name) != value
    ]
    if reflection.should_adjust and changed:
        reason = (
            "候选模型误差未优于真实基线；仅将已有配置提升到下一个白名单档位。"
        )
        confidence = 0.85
    elif reflection.performance_analysis.status == "stable":
        reason = "现有实验未显示相对基线的性能问题，保持已验证配置。"
        confidence = 0.9
    elif result.status != ExperimentStatus.COMPLETED:
        reason = "实验未完成，禁止依据不完整结果调整参数。"
        confidence = 1.0
    else:
        reason = "没有可比较的候选与基线误差，保持配置并禁止猜测参数。"
        confidence = 0.4
    return ExperimentOptimizationSuggestion(
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        current_configuration=current,
        performance_analysis=reflection.performance_analysis,
        next_configuration=next_configuration,
        reason=reason,
        confidence=confidence,
        changed_parameters=changed,
    )


def build_next_experiment_contract(
    contract: ExperimentContract,
    suggestion: ExperimentOptimizationSuggestion,
    *,
    next_experiment_id: str,
    next_plan_id: str,
) -> ExperimentContract:
    """Explicitly create a next contract while preserving frozen test semantics."""
    if suggestion.experiment_id != contract.experiment_id:
        raise ValueError("suggestion_experiment_id_mismatch")
    if suggestion.hypothesis_id != contract.hypothesis_id:
        raise ValueError("suggestion_hypothesis_id_mismatch")
    if not next_experiment_id or next_experiment_id == contract.experiment_id:
        raise ValueError("distinct_next_experiment_id_required")
    if not next_plan_id:
        raise ValueError("next_plan_id_required")

    payload = contract.model_dump(mode="python")
    next_config = suggestion.next_configuration.model_dump(exclude_none=True)
    if "window_size" in next_config:
        payload["window_steps"] = next_config["window_size"]
    if "forecast_horizon" in next_config:
        interval = contract.sampling_interval_seconds
        if interval is None or interval <= 0:
            raise ValueError("sampling_interval_required_for_forecast_horizon")
        seconds = next_config["forecast_horizon"] * 60
        if seconds % interval:
            raise ValueError("forecast_horizon_not_aligned_to_sampling_interval")
        payload["prediction_horizon_steps"] = seconds // interval

    requirements = dict(contract.execution_requirements)
    model_parameters = {
        key: next_config[key]
        for key in ("d_model", "hidden_size", "learning_rate")
        if key in next_config
    }
    if model_parameters:
        requirements["model_configuration"] = model_parameters
    requirements["optimization_source_experiment_id"] = contract.experiment_id
    payload.update({
        "experiment_id": next_experiment_id,
        "plan_id": next_plan_id,
        "status": ExperimentStatus.PLANNED,
        "execution_requirements": requirements,
        "optimization_suggestion": suggestion.model_dump(mode="json"),
    })
    return ExperimentContract.model_validate(payload)


def optimization_contract_issues(
    contract: ExperimentContract,
    *,
    capability,
    model_registry,
) -> list[str]:
    """Use the existing registries to fail closed before scheduling a contract."""
    issues: list[str] = []
    if contract.optimization_suggestion is None:
        issues.append("optimization_suggestion_required")
    else:
        try:
            suggestion = ExperimentOptimizationSuggestion.model_validate(
                contract.optimization_suggestion
            )
            runner_unsupported = sorted(
                set(suggestion.changed_parameters)
                & {"d_model", "hidden_size", "learning_rate"}
            )
            for parameter in runner_unsupported:
                issues.append(
                    f"runner_parameter_override_not_supported:{parameter}"
                )
        except Exception as exc:
            issues.append(f"invalid_optimization_suggestion:{exc}")

    if contract.window_steps != capability.window_steps:
        issues.append(
            f"unsupported_window_size:{contract.window_steps}"
        )
    if (
        contract.prediction_horizon_steps
        != capability.prediction_horizon_steps_value()
    ):
        issues.append(
            "unsupported_prediction_horizon_steps:"
            f"{contract.prediction_horizon_steps}"
        )
    if (
        contract.sampling_interval_seconds
        != capability.sampling_interval_seconds_value()
    ):
        issues.append(
            "unsupported_sampling_interval_seconds:"
            f"{contract.sampling_interval_seconds}"
        )

    match = capability.check_executable(
        required_operations=contract.required_operations,
        required_models=contract.candidate_models,
        required_metrics=contract.metrics,
        required_variables=[
            *contract.input_variables,
            contract.target_variable,
        ],
        requires_feature_intervention=(
            contract.experiment_type
            in {"feature_ablation", "feature_intervention"}
        ),
    )
    issues.extend(match.missing_capabilities)

    compatible = {
        spec.model_name.casefold()
        for spec in model_registry.compatible_with_capability(
            capability,
            target=contract.target_variable,
            metrics=contract.metrics,
            window_steps=contract.window_steps,
            horizon_steps=contract.prediction_horizon_steps,
            sampling_interval_seconds=contract.sampling_interval_seconds,
        )
    }
    for model_name in contract.candidate_models:
        if model_name.casefold() not in compatible:
            issues.append(f"model_contract_incompatible:{model_name}")
            continue
        spec = model_registry.get(model_name)
        if spec.required_features != len(contract.input_variables):
            issues.append(f"model_feature_count_mismatch:{model_name}")

    return list(dict.fromkeys(issues))
