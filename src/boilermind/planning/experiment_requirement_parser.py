from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field


EXPERIMENT_TYPE_FEATURE_INTERVENTION = (
    "feature_intervention"
)

EXPERIMENT_TYPE_CONSTRAINED_OPTIMIZATION = (
    "constrained_optimization"
)

EXPERIMENT_TYPE_MULTI_VARIABLE_INTERVENTION = (
    "multi_variable_intervention"
)

EXPERIMENT_TYPE_MULTI_TARGET_FORECAST = (
    "multi_target_forecast"
)

EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON = (
    "reference_model_comparison"
)

EXPERIMENT_TYPE_MODEL_COMPARISON = (
    "model_comparison"
)

EXPERIMENT_TYPE_LOCKED_TEST_EVALUATION = (
    "locked_test_evaluation"
)

EXPERIMENT_TYPE_DIRECTION_RATE_INTERACTION = (
    "direction_rate_interaction_evaluation"
)

EXPERIMENT_TYPE_SENSOR_CORRUPTION = (
    "sensor_corruption_robustness_evaluation"
)

EXPERIMENT_TYPE_UNSUPPORTED_SCIENTIFIC_DESIGN = (
    "unsupported_scientific_design"
)


class ExperimentRequirements(BaseModel):
    """
    Required experiment capabilities parsed from ONE
    QualifiedHypothesis.

    This is a pure extraction result. Executability is
    decided later by ExperimentCapabilityRegistry.
    """

    hypothesis_id: str = Field(min_length=1)

    experiment_type: str = Field(min_length=1)

    required_operations: list[str] = Field(
        min_length=1,
    )

    required_models: list[str] = Field(
        default_factory=list,
    )

    required_model_roles: dict[str, str] = Field(
        default_factory=dict,
    )

    required_targets: list[str] = Field(
        default_factory=list,
    )

    required_metrics: list[str] = Field(
        default_factory=list,
    )

    required_variables: list[str] = Field(
        default_factory=list,
    )

    prediction_horizon_steps: int | None = None

    control: dict[str, Any] = Field(
        default_factory=dict,
    )

    treatment: dict[str, Any] = Field(
        default_factory=dict,
    )

    hard_constraints: list[str] = Field(
        default_factory=list,
    )

    confirmation_criteria: list[str] = Field(
        min_length=1,
    )

    falsification_criteria: list[str] = Field(
        min_length=1,
    )

    requires_feature_intervention: bool = False


class FrozenHypothesisDesign(BaseModel):
    """Deterministic scientific meaning frozen before planning."""

    schema_version: str = "boilermind.hypothesis_design.v2"
    experiment_type: str = Field(min_length=1)
    required_operations: list[str] = Field(default_factory=list)
    required_models: list[str] = Field(default_factory=list)
    required_model_roles: dict[str, str] = Field(default_factory=dict)
    required_targets: list[str] = Field(default_factory=list)
    required_metrics: list[str] = Field(default_factory=list)
    required_variables: list[str] = Field(default_factory=list)
    prediction_horizon_steps: int | None = None
    control: dict[str, Any] = Field(default_factory=dict)
    treatment: dict[str, Any] = Field(default_factory=dict)
    confirmation_criteria: list[str] = Field(default_factory=list)
    falsification_criteria: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    requires_feature_intervention: bool = False


def freeze_hypothesis_design(
    hypothesis: dict[str, Any],
) -> FrozenHypothesisDesign:
    requirements = parse_hypothesis_requirements(hypothesis)
    return FrozenHypothesisDesign(
        experiment_type=requirements.experiment_type,
        required_operations=sorted(set(requirements.required_operations)),
        required_models=sorted(set(requirements.required_models)),
        required_model_roles=dict(sorted(requirements.required_model_roles.items())),
        required_targets=sorted(set(requirements.required_targets)),
        required_metrics=list(dict.fromkeys(requirements.required_metrics)),
        required_variables=sorted(set(requirements.required_variables)),
        prediction_horizon_steps=requirements.prediction_horizon_steps,
        control=requirements.control,
        treatment=requirements.treatment,
        confirmation_criteria=list(requirements.confirmation_criteria),
        falsification_criteria=list(requirements.falsification_criteria),
        hard_constraints=list(requirements.hard_constraints),
        requires_feature_intervention=requirements.requires_feature_intervention,
    )


def frozen_design_sha256(
    design: FrozenHypothesisDesign | dict[str, Any],
) -> str:
    validated = (
        design if isinstance(design, FrozenHypothesisDesign)
        else FrozenHypothesisDesign.model_validate(design)
    )
    payload = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def requirements_from_frozen_design(
    hypothesis_id: str,
    design: FrozenHypothesisDesign | dict[str, Any],
) -> ExperimentRequirements:
    frozen = (
        design if isinstance(design, FrozenHypothesisDesign)
        else FrozenHypothesisDesign.model_validate(design)
    )
    return ExperimentRequirements(
        hypothesis_id=hypothesis_id,
        experiment_type=frozen.experiment_type,
        required_operations=list(frozen.required_operations),
        required_models=list(frozen.required_models),
        required_model_roles=dict(frozen.required_model_roles),
        required_targets=list(frozen.required_targets),
        required_metrics=list(frozen.required_metrics),
        required_variables=list(frozen.required_variables),
        prediction_horizon_steps=frozen.prediction_horizon_steps,
        control=dict(frozen.control),
        treatment=dict(frozen.treatment),
        hard_constraints=list(frozen.hard_constraints),
        confirmation_criteria=list(frozen.confirmation_criteria),
        falsification_criteria=list(frozen.falsification_criteria),
        requires_feature_intervention=frozen.requires_feature_intervention,
    )


def frozen_design_alignment_issues(
    hypothesis: dict[str, Any],
) -> list[str]:
    frozen_payload = hypothesis.get("scientific_design")
    if frozen_payload is None:
        return []
    try:
        frozen = FrozenHypothesisDesign.model_validate(frozen_payload)
    except Exception as exc:
        return [f"hypothesis_design:invalid:{exc}"]
    current = freeze_hypothesis_design({
        key: value for key, value in hypothesis.items()
        if key != "scientific_design"
    })
    if frozen.model_dump(mode="json") == current.model_dump(mode="json"):
        return []
    issues: list[str] = []
    old = frozen.model_dump(mode="json")
    new = current.model_dump(mode="json")
    for key in old:
        if old[key] != new[key]:
            issues.append(f"hypothesis_design:semantic_drift:{key}")
    return issues


_MODEL_PATTERNS: list[tuple[str, str]] = [
    ("bayesianridge", r"bayesian[\s_-]?ridge"),
    ("ridge", r"(?<![a-z0-9])ridge(?![a-z0-9])"),
    ("hgb", r"(?<![a-z0-9])hgb(?![a-z0-9])"),
    ("svr", r"(?<![a-z0-9])svr(?![a-z0-9])"),
    ("rf", r"(?<![a-z0-9])rf(?![a-z0-9])"),
    ("mlp", r"(?<![a-z0-9])mlp(?![a-z0-9])"),
    ("elasticnet", r"elastic[\s_-]?net"),
    ("pls", r"(?<![a-z0-9])pls(?![a-z0-9])"),
    ("knn", r"(?<![a-z0-9])knn(?![a-z0-9])"),
    ("gpr", r"(?<![a-z0-9])gpr(?![a-z0-9])"),
    ("lstm", r"(?<![a-z0-9])lstm(?![a-z0-9])"),
    ("transformer", r"transformer"),
    ("tcn", r"(?<![a-z0-9])tcn(?![a-z0-9])"),
    ("dlinear", r"d[\s_-]?linear"),
    ("gru", r"(?<![a-z0-9])gru(?![a-z0-9])"),
    ("patchtst", r"patchtst"),
    ("itransformer", r"itransformer"),
    ("timesnet", r"timesnet"),
    ("persistence", r"persistence"),
]


_FEATURE_INTERVENTION_KEYWORDS = [
    "lag",
    "滞后",
    "时延",
    "时间延迟",
    "延迟处理",
    "消融",
    "ablation",
    "feature_intervention",
    "特征干预",
]


_CONSTRAINED_KEYWORDS = [
    "约束优化",
    "constrained_optimization",
    "constrained optimization",
    "压力约束",
    "pressure constraint",
    "优化求解",
    "寻优",
    "上限",
    "不得超过",
    "不超过",
]


_MULTI_VARIABLE_KEYWORDS = [
    "multi_variable_intervention",
    "multi-variable intervention",
    "多变量干预",
    "给煤量",
    "给水流量",
]


_MULTI_TARGET_KEYWORDS = [
    "multi_target_forecast",
    "multi-target forecast",
    "多目标",
]


_METRIC_KEYWORDS = [
    ("MAE", ["mae", "平均绝对误差", "绝对误差"]),
    ("RMSE", ["rmse", "均方根误差", "均方误差"]),
    ("R2", ["r2", "r²", "决定系数"]),
    ("MBE", ["mbe", "平均偏差"]),
    ("MAPE", ["mape", "平均绝对百分比误差"]),
    ("IQR", ["iqr", "四分位距", "四分位"]),
]


def _model_names(text: str) -> list[str]:
    lowered = text.lower()
    found = []

    for model_id, pattern in _MODEL_PATTERNS:
        if re.search(pattern, lowered):
            found.append(model_id)

    return list(dict.fromkeys(found))


def _metrics(text: str) -> list[str]:
    lowered = text.lower()
    found = []

    for metric_id, keywords in _METRIC_KEYWORDS:
        if any(
            keyword in lowered
            for keyword in keywords
        ):
            found.append(metric_id)

    return list(dict.fromkeys(found))


def _horizon_steps(text: str) -> int | None:
    lowered = text.lower()

    explicit = re.search(
        r"(?<![a-z0-9_])h\s*(80|40|20|8|4)(?![0-9])",
        lowered,
    )
    if explicit:
        return int(explicit.group(1))

    if (
        "10分钟" in lowered
        or "10 分钟" in lowered
        or "40步" in lowered
    ):
        return 40

    if (
        "5分钟" in lowered
        or "5 分钟" in lowered
    ):
        return 20

    if (
        "20分钟" in lowered
        or "20 分钟" in lowered
    ):
        return 80

    if (
        "1分钟" in lowered
        or "1 分钟" in lowered
    ):
        return 4

    if (
        "2分钟" in lowered
        or "2 分钟" in lowered
    ):
        return 8

    return None


def _targets(text: str) -> list[str]:
    lowered = text.lower()
    targets = []

    if any(
        keyword in lowered
        for keyword in [
            "主蒸汽流量",
            "质量流量",
            "mass flow",
            "main_steam_mass_flow",
            "t/h",
        ]
    ):
        targets.append(
            "main_steam_mass_flow"
        )

    if any(
        keyword in lowered
        for keyword in [
            "蒸汽体积流量",
            "体积流量",
            "volumetric",
            "steam_volumetric_flow",
            "m3/s",
            "m³/s",
        ]
    ):
        targets.append(
            "steam_volumetric_flow"
        )

    return list(dict.fromkeys(targets))


def _hard_constraints(text: str) -> list[str]:
    lowered = text.lower()
    constraints = []

    for keyword in [
        "压力约束",
        "pressure constraint",
        "上限",
        "不得超过",
        "不超过",
        "must not exceed",
        "<=",
        "≤",
    ]:
        if keyword in lowered:
            constraints.append(
                "hard_constraint_detected:"
                f"{keyword}"
            )

    return list(dict.fromkeys(constraints))


def _criteria(
    experiment_type: str,
    metrics: list[str],
    hypothesis: dict[str, Any],
) -> tuple[list[str], list[str]]:

    if hypothesis.get(
        "confirmation_criteria"
    ) and hypothesis.get(
        "falsification_criteria"
    ):
        return (
            [
                str(item)
                for item in hypothesis[
                    "confirmation_criteria"
                ]
            ],
            [
                str(item)
                for item in hypothesis[
                    "falsification_criteria"
                ]
            ],
        )

    core_metrics = [
        metric
        for metric in metrics
        if metric in {"MAE", "RMSE", "R2", "MBE"}
    ]

    metric_list = (
        ",".join(core_metrics)
        if core_metrics
        else "MAE"
    )

    text = str(
        hypothesis.get("hypothesis") or hypothesis.get("statement") or ""
    ).casefold()
    if "相比" in text:
        left_text, right_text = text.split("相比", 1)
        left_models = _model_names(left_text)
        right_models = _model_names(right_text)
        if left_models and right_models:
            reference = right_models[0]
            compared = [model for model in left_models if model != reference]
            if compared and any(token in right_text for token in ("不能", "并不", "不优于", "没有")):
                payload = f"{','.join(compared)}|{reference}|{metric_list}"
                return (
                    ["all_models_not_better_than_model_on:" + payload],
                    ["any_model_better_than_model_on:" + payload],
                )

    lower_match = re.search(r"(.{0,80}?)(?:性能|精度|mae|r2)?(?:将|会)?低于(.{0,120})", text)
    if lower_match:
        lower_error_models = _model_names(lower_match.group(1))
        reference_models = _model_names(lower_match.group(2))
        if lower_error_models and reference_models:
            payload = (
                f"{','.join(lower_error_models)}|{reference_models[0]}|{metric_list}"
            )
            return (
                ["all_models_better_than_model_on:" + payload],
                ["any_model_not_better_than_model_on:" + payload],
            )

    if (
        experiment_type
        == EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON
    ):
        return (
            [
                "all_candidates_worse_than_reference_on:"
                + metric_list,
            ],
            [
                "any_candidate_better_than_reference_on:"
                + metric_list,
            ],
        )

    if (
        experiment_type
        == EXPERIMENT_TYPE_FEATURE_INTERVENTION
    ):
        return (
            [
                "any_model_with_intervention_better_"
                "than_control_on:"
                + metric_list,
            ],
            [
                "no_model_with_intervention_better_on:"
                + metric_list,
            ],
        )

    if (
        experiment_type
        == EXPERIMENT_TYPE_MODEL_COMPARISON
    ):
        return (
            [
                "candidate_better_than_baseline_on:"
                + metric_list,
            ],
            [
                "candidate_not_better_than_baseline_on:"
                + metric_list,
            ],
        )

    return (
        ["predeclared_target_achieved"],
        ["predeclared_target_not_achieved"],
    )


def _control_treatment(
    experiment_type: str,
    models: list[str],
    reference_models: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:

    if (
        experiment_type
        == EXPERIMENT_TYPE_FEATURE_INTERVENTION
    ):
        return (
            {
                "role": "control",
                "feature_set": "no_lag_features",
                "description": (
                    "不使用滞后/时延特征输入"
                ),
            },
            {
                "role": "treatment",
                "feature_set": "with_lag_features",
                "description": (
                    "引入滞后/时延特征输入"
                ),
            },
        )

    if (
        experiment_type
        == EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON
    ):
        return (
            {
                "role": "reference",
                "models": list(reference_models),
                "description": (
                    "reference model 基线评价"
                ),
            },
            {
                "role": "treatment",
                "models": [
                    model
                    for model in models
                    if model not in reference_models
                ],
                "description": (
                    "预声明候选模型在锁定测试集上的评价"
                ),
            },
        )

    if (
        experiment_type
        == EXPERIMENT_TYPE_MODEL_COMPARISON
    ):
        return (
            {
                "role": "baseline_model",
                "models": list(reference_models),
                "description": "对照模型",
            },
            {
                "role": "candidate_model",
                "models": [
                    model
                    for model in models
                    if model not in reference_models
                ],
                "description": "候选模型",
            },
        )

    return (
        {
            "role": "control",
            "description": "当前输入 / 无干预",
        },
        {
            "role": "treatment",
            "description": (
                "施加实验干预并满足预声明约束"
            ),
        },
    )


def parse_hypothesis_requirements(
    hypothesis: dict[str, Any],
) -> ExperimentRequirements:
    """
    Deterministically extract required experiment capabilities
    from ONE QualifiedHypothesis dict.

    Raises ValueError when the hypothesis cannot be parsed
    (missing id / statement). There is NO fallback.
    """

    if not isinstance(hypothesis, dict):
        raise ValueError(
            "hypothesis_dict_required"
        )

    hypothesis_id = (
        hypothesis.get("hypothesis_id")
        or hypothesis.get("id")
    )

    if not hypothesis_id:
        raise ValueError(
            "hypothesis_id_required"
        )

    statement = (
        hypothesis.get("hypothesis")
        or hypothesis.get("statement")
    )

    if not statement:
        raise ValueError(
            "hypothesis_statement_required"
        )

    verification_intent = str(
        hypothesis.get("verification_intent", "")
    )

    falsification_condition = str(
        hypothesis.get("falsification_condition", "")
    )

    text = (
        f"{statement} {verification_intent} "
        f"{falsification_condition}"
    ).lower()

    # ------------------------------------------------
    # experiment_type (most specific first)
    # ------------------------------------------------

    direction_terms = (
        "升负荷", "降负荷", "升降负荷", "负荷方向",
        "ramp_up", "ramp_down", "load direction",
    )
    rate_terms = (
        "变化速率", "变负荷速率", "负荷速率",
        "load change rate", "ramp rate",
    )
    corruption_terms = (
        "高斯噪声", "噪声注入", "噪声水平", "噪声强度",
        "传感器噪声", "毛刺", "尖峰", "漂移", "冻结",
        "缺失注入", "污染比例", "gaussian noise", "noise injection",
        "noise level", "spike injection", "sensor corruption",
    )
    manipulation_terms = (
        "注入", "不同水平", "不同强度", "随", "增加", "扫描",
        "injection", "level", "sweep", "corruption",
    )

    if (
        any(keyword in text for keyword in corruption_terms)
        and any(keyword in text for keyword in manipulation_terms)
    ):
        experiment_type = EXPERIMENT_TYPE_SENSOR_CORRUPTION

    elif (
        any(keyword in text for keyword in direction_terms)
        and any(keyword in text for keyword in rate_terms)
    ):
        experiment_type = (
            EXPERIMENT_TYPE_DIRECTION_RATE_INTERACTION
        )

    elif any(
        keyword in text
        for keyword in _FEATURE_INTERVENTION_KEYWORDS
    ):
        experiment_type = (
            EXPERIMENT_TYPE_FEATURE_INTERVENTION
        )

    elif any(
        keyword in text
        for keyword in _CONSTRAINED_KEYWORDS
    ):
        experiment_type = (
            EXPERIMENT_TYPE_CONSTRAINED_OPTIMIZATION
        )

    elif any(
        keyword in text
        for keyword in _MULTI_VARIABLE_KEYWORDS
    ):
        experiment_type = (
            EXPERIMENT_TYPE_MULTI_VARIABLE_INTERVENTION
        )

    elif any(
        keyword in text
        for keyword in _MULTI_TARGET_KEYWORDS
    ):
        experiment_type = (
            EXPERIMENT_TYPE_MULTI_TARGET_FORECAST
        )

    elif (
        "reference_model_comparison" in text
        or "与persistence" in text
        or "与 persistence" in text
        or "persistence模型" in text
        or "参考模型" in text
    ):
        experiment_type = (
            EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON
        )

    elif (
        re.search(
            r"(?<![a-z_])model_comparison(?![a-z_])",
            text,
        )
        or "模型对比" in text
        or "比较不同模型" in text
    ):
        experiment_type = (
            EXPERIMENT_TYPE_MODEL_COMPARISON
        )

    elif any(
        keyword in text
        for keyword in (
            "locked_test", "锁定测试", "独立测试",
            "holdout", "留出测试",
        )
    ):
        experiment_type = (
            EXPERIMENT_TYPE_LOCKED_TEST_EVALUATION
        )

    else:
        # An unknown scientific design must never silently become a
        # generic locked-test model comparison.  That substitution can
        # produce a real experiment which does not test the hypothesis.
        experiment_type = (
            EXPERIMENT_TYPE_UNSUPPORTED_SCIENTIFIC_DESIGN
        )

    # ------------------------------------------------
    # required_operations
    # ------------------------------------------------

    required_operations = set()

    if experiment_type == EXPERIMENT_TYPE_SENSOR_CORRUPTION:
        required_operations.update({
            "sensor_corruption_injection",
            "corruption_level_sweep",
            "clean_corrupted_paired_comparison",
            "robustness_degradation_evaluation",
        })
        if any(token in text for token in (
            "高斯噪声", "噪声注入", "gaussian noise", "noise injection",
        )):
            required_operations.add("gaussian_noise_injection")
        if any(token in text for token in (
            "毛刺", "尖峰", "spike injection",
        )):
            required_operations.add("spike_injection")
        if "漂移" in text:
            required_operations.add("drift_injection")
        if "冻结" in text:
            required_operations.add("freeze_injection")
        if any(token in text for token in ("缺失", "missingness")):
            required_operations.add("missingness_injection")

    if (
        experiment_type
        == EXPERIMENT_TYPE_DIRECTION_RATE_INTERACTION
    ):
        required_operations.update({
            "load_rate_computation",
            "direction_regime_assignment",
            "rate_stratified_evaluation",
            "direction_rate_interaction_evaluation",
        })

    if (
        experiment_type
        == EXPERIMENT_TYPE_UNSUPPORTED_SCIENTIFIC_DESIGN
    ):
        required_operations.add(
            "unsupported_scientific_design"
        )

    if (
        "chronological" in text
        or "chronological_validation" in text
        or "时间顺序" in text
    ):
        required_operations.add(
            "chronological_validation"
        )

    if (
        "locked_test" in text
        or "锁定测试" in text
    ):
        required_operations.add(
            "locked_test_evaluation"
        )

    if (
        "reference_model_comparison" in text
        or "persistence" in text
        or "与persistence" in text
    ):
        required_operations.add(
            "reference_model_comparison"
        )

    if (
        re.search(
            r"(?<![a-z_])model_comparison(?![a-z_])",
            text,
        )
        or "模型对比" in text
        or "比较不同模型" in text
    ):
        required_operations.add(
            "model_comparison"
        )

    requires_feature_intervention = any(
        keyword in text
        for keyword in _FEATURE_INTERVENTION_KEYWORDS
    )

    if requires_feature_intervention:
        required_operations.add(
            "feature_intervention"
        )

    if any(
        keyword in text
        for keyword in _MULTI_VARIABLE_KEYWORDS
    ):
        required_operations.add(
            "multi_variable_intervention"
        )

    if any(
        keyword in text
        for keyword in _MULTI_TARGET_KEYWORDS
    ):
        required_operations.add(
            "multi_target_forecast"
        )

    if any(
        keyword in text
        for keyword in _CONSTRAINED_KEYWORDS
    ):
        required_operations.add(
            "constrained_optimization"
        )

        required_operations.add(
            "hard_constraint_evaluation"
        )

    if any(
        keyword in text
        for keyword in (
            "工况分层", "工况细分", "不同运行工况", "regime (",
            "ramp_up", "ramp_down",
            "direction_change", "steady",
        )
    ):
        required_operations.add("regime_stratified_evaluation")

    if any(keyword in text for keyword in ("显著", "统计显著", "significant")):
        required_operations.add("statistical_significance_evaluation")

    if any(keyword in text for keyword in ("多随机种子", "多个种子", "multi-seed", "multi seed")):
        required_operations.add("multi_seed_evaluation")

    # ------------------------------------------------
    # models / roles / metrics / targets / horizon
    # ------------------------------------------------

    models = _model_names(text)

    # A single hypothesis may not hide several independently falsifiable
    # pairwise comparisons behind one "lower than A and B" sentence. The
    # criterion compiler can freeze only one reference relation at a time.
    lower_relation = re.search(
        r"([^。；;\n]{0,80}?)(?:性能|精度|mae|r2)?(?:将|会)?低于"
        r"([^。；;\n]{0,120})",
        text,
    )
    if lower_relation and len(_model_names(lower_relation.group(2))) > 1:
        required_operations.add("single_reference_relation_required")

    # Model identities provide a deterministic design signal even when the
    # prose does not use one of the preferred operation phrases.  Resolve
    # this before criteria compilation, while keeping genuinely unknown
    # scientific relations fail-closed.
    if (
        experiment_type
        == EXPERIMENT_TYPE_UNSUPPORTED_SCIENTIFIC_DESIGN
    ):
        if "persistence" in models and len(models) >= 2:
            experiment_type = (
                EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON
            )
            required_operations.discard(
                "unsupported_scientific_design"
            )
            required_operations.add(
                "reference_model_comparison"
            )
        elif len(models) >= 2:
            experiment_type = EXPERIMENT_TYPE_MODEL_COMPARISON
            required_operations.discard(
                "unsupported_scientific_design"
            )
            required_operations.add("model_comparison")

    if len(models) >= 2:
        required_operations.add("model_comparison")

    reference_models = []

    if "persistence" in models:
        reference_models.append(
            "persistence"
        )

    model_roles = {
        model: (
            "reference"
            if model in reference_models
            else "candidate"
        )
        for model in models
    }

    metrics = _metrics(text)

    targets = _targets(text)

    metrics = (
        metrics
        or ["MAE"]
    )

    primary_metric = metrics[0]
    secondary_metrics = metrics[1:]

    hard_constraints = _hard_constraints(text)

    confirmation_criteria, falsification_criteria = (
        _criteria(
            experiment_type,
            metrics,
            hypothesis,
        )
    )

    control, treatment = _control_treatment(
        experiment_type,
        models,
        reference_models,
    )

    required_variables = [
        str(item)
        for item in (
            hypothesis.get("variables")
            or hypothesis.get(
                "related_variables",
                [],
            )
            or []
        )
    ]
    metric_names = {item.casefold() for item in metrics}
    non_sensor_fields = {
        "prediction_horizon_steps", "horizon", "forecast_horizon",
        "regime_definition", "experiment_type", "model_id",
        "steam_volumetric_flow", "main_steam_mass_flow",
        *(item.casefold() for item in targets),
        *(item.casefold() for item in models),
    }
    required_variables = [
        item for item in required_variables
        if item.strip().casefold() not in metric_names
        and item.strip().casefold() not in non_sensor_fields
    ]

    declared_variables = {
        str(item).strip().casefold()
        for item in (
            hypothesis.get("variables")
            or hypothesis.get("related_variables", [])
            or []
        )
    }
    if "regime_definition" in declared_variables:
        required_operations.add("regime_stratified_evaluation")

    if not required_operations:
        required_operations.add(
            {
                EXPERIMENT_TYPE_REFERENCE_MODEL_COMPARISON:
                    "reference_model_comparison",
                EXPERIMENT_TYPE_LOCKED_TEST_EVALUATION:
                    "locked_test_evaluation",
            }.get(experiment_type, "model_comparison")
        )

    return ExperimentRequirements(
        hypothesis_id=str(hypothesis_id),
        experiment_type=experiment_type,
        required_operations=sorted(
            required_operations
        ),
        required_models=models,
        required_model_roles=model_roles,
        required_targets=targets,
        required_metrics=metrics,
        required_variables=required_variables,
        prediction_horizon_steps=(
            _horizon_steps(text)
        ),
        control=control,
        treatment=treatment,
        hard_constraints=hard_constraints,
        confirmation_criteria=(
            confirmation_criteria
        ),
        falsification_criteria=(
            falsification_criteria
        ),
        requires_feature_intervention=(
            requires_feature_intervention
        ),
    )
