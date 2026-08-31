"""Deterministically assemble a post-experiment scientific research plan."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from boilermind.core.contracts.base import ContractModel
from boilermind.core.contracts.evidence import EvidenceBundle
from boilermind.core.contracts.experiment import (
    ExperimentAudit, ExperimentContract, ExperimentPlan, ExperimentResult,
    ScientificResult,
)
from boilermind.core.contracts.hypothesis import ScientificHypothesis
from boilermind.core.contracts.research_problem import ResearchProblemSpec
from boilermind.core.contracts.scientific_research_plan import (
    DatasetSection, ExperimentValiditySnapshot, ExperimentsSection,
    MethodsSection, MetricsSection, ModelResultSnapshot,
    ProblemStatementSection, ProvenanceEntry, RationaleSection,
    ReferenceEntry, ResearchTraceEntry, ResultsSection,
    ScientificResearchPlan, ScientificResearchPlanMetadata,
    ScientificVerdictSnapshot, TechnicalDetailsSection,
    FinalPlanSelection, PaperAbstractSection,
)


class ScientificResearchPlanGeneratorInput(ContractModel):
    research_problem: ResearchProblemSpec
    evidence_bundle: EvidenceBundle | None = None
    hypothesis: ScientificHypothesis
    experiment_plan: ExperimentPlan
    experiment_contract: ExperimentContract
    experiment_result: ExperimentResult
    experiment_audit: ExperimentAudit
    scientific_result: ScientificResult
    research_trace: list[dict[str, Any]]
    run_id: str | None = None
    round_index: int = 1
    revision_index: int = 0
    selection_reason: str = "FIRST_ROUND_NO_ITERATION"
    iteration_occurred: bool = False
    fallback_applied: bool = False


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _sample_counts(record: Any) -> dict[str, int] | None:
    values = {
        "train": record.train_samples,
        "validation": record.validation_samples,
        "test": record.test_samples,
    }
    counts = {name: value for name, value in values.items() if value is not None}
    return counts or None


def _dataset_counts(result: ExperimentResult) -> dict[str, int] | None:
    observed = [_sample_counts(record) for record in result.model_records.values()]
    observed = [counts for counts in observed if counts is not None]
    if not observed or any(counts != observed[0] for counts in observed[1:]):
        return None
    return observed[0]


def _metric_value(metrics: dict[str, Any] | None, name: str) -> float | None:
    wanted = str(name).casefold()
    for key, value in (metrics or {}).items():
        if str(key).casefold() == wanted:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _best_model(
    metrics_by_model: dict[str, dict[str, Any]], metric: str,
) -> str | None:
    observed = {
        name: value
        for name, values in metrics_by_model.items()
        if (value := _metric_value(values, metric)) is not None
    }
    if not observed:
        return None
    chooser = max if metric.casefold() == "r2" else min
    return chooser(observed, key=observed.get)


def _display_model(name: str | None) -> str:
    return {
        "hgb_control_optimizer": "HGB 控制优化模型",
        "current_operating_point": "当前工况基线",
        "persistence": "持久性基线模型",
    }.get(str(name or "").casefold(), str(name or "未形成"))


def _display_target(name: str | None) -> str:
    return {
        "steam_volumetric_flow": "蒸汽体积流量 V",
        "main_steam_volumetric_flow": "主蒸汽体积流量 V",
    }.get(str(name or "").casefold(), str(name or "目标变量"))


def _display_verdict(value: Any) -> str:
    return {"supported": "得到支持", "partially_supported": "部分支持",
            "insufficient_evidence": "证据不足", "falsified": "未获支持"}.get(
        _enum_value(value).casefold(), _enum_value(value)
    )


def _model_hypothesis(
    model: str,
    target: str,
    horizon: int | None,
    *,
    window_steps: int | None,
    sampling_interval_seconds: int | None,
    primary_metric: str,
    secondary_metrics: list[str],
    random_seed: int | None,
) -> dict[str, Any]:
    model_label = _display_model(model)
    target_label = _display_target(target)
    horizon_label = "当前时刻" if horizon is None else f"{horizon} 步"
    descriptions = {
        "ridge": (
            "正则化线性稳定性假设",
            "若目标与输入之间以近似线性、共线关系为主，Ridge 的 L2 正则化应降低参数方差，"
            "并在时间外推验证中保持稳定误差。",
            "验证集主指标优于参考模型，且锁定测试集不出现明显退化。",
        ),
        "bayesianridge": (
            "贝叶斯收缩稳定性假设",
            "若样本存在共线性与参数不确定性，BayesianRidge 的概率收缩应比无学习参考模型"
            "获得更稳定的验证集表现。",
            "验证集主指标达到候选最优或接近最优，并在锁定测试集保持优势。",
        ),
        "rf": (
            "非线性交互捕捉假设",
            "若锅炉多变量对目标存在阈值效应和非线性交互，RandomForest 应降低 locked-test"
            "平均绝对误差；但其跨时间块稳定性仍需单独复验。",
            "相对线性候选与 Persistence 呈现可重复的误差改善。",
        ),
        "persistence": (
            "惯性参考假设",
            "若未来目标主要由短期惯性支配，Persistence 将形成具有竞争力的强参考；"
            "候选模型只有稳定超过它才具有新增信息价值。",
            "候选模型未能稳定降低参考模型误差。",
        ),
        "hgb_control_optimizer": (
            "梯度提升软测控制优化假设",
            "若给煤、给水、送风与汽包压力对蒸汽体积量存在可学习的非线性响应，"
            "HistGradientBoosting 软测模型应在时间外推验证中保持低 MAE，"
            "并由约束搜索给出满足压力上限与提升目标的可行调参方案。",
            "验证 MAE 足够低，且约束搜索找到满足压力≤23MPa 与 V 提升 15% 的可行候选。",
        ),
    }
    title, statement, expected = descriptions.get(model.casefold(), (
        f"{model_label} 适用性假设",
        f"若 {model_label} 的结构与当前数据特征匹配，其对 {target_label} 的预测表现应优于参考模型。",
        "模型在验证集和锁定测试集上的表现保持一致。",
    ))
    key = model.casefold()
    configuration = {
        "ridge": "冻结 L2 正则化线性回归配置；所有超参数仅可由训练集/validation 确定。",
        "bayesianridge": "冻结 BayesianRidge 先验与收敛配置；记录系数及不确定性相关参数快照。",
        "rf": "冻结树数量、深度、叶节点样本数与特征抽样配置；固定随机种子。",
        "persistence": "无训练参数；以预测起点最近一次可观测目标值作为未来时域预测。",
        "hgb_control_optimizer": (
            "HistGradientBoostingRegressor（max_depth=6，max_iter=200，"
            "learning_rate=0.1，random_seed=0）；仅在训练集拟合，验证集只评估一次。"
        ),
    }.get(key, "使用能力注册表中的冻结模型配置，并保存完整参数快照。")
    if key == "persistence":
        confirmation_rule = (
            f"若所有学习模型在锁定测试集的 {primary_metric} 均未低于持久性基线模型，支持惯性参考假设。"
        )
        falsification_rule = (
            f"若至少一个学习模型在锁定测试集稳定降低 {primary_metric}，则不支持强惯性充分性。"
        )
    else:
        confirmation_rule = (
            f"{model_label} 在验证集按 {primary_metric} 参与比较，且锁定测试集的 {primary_metric} "
            "低于持久性基线模型。"
        )
        falsification_rule = (
            f"{model_label} 在锁定测试集的 {primary_metric} 不低于持久性基线模型，或跨时间块优势不可重复。"
        )
    return {
        "model": model,
        "title": title,
        "statement": statement,
        "expected_observation": expected,
        "horizon_steps": horizon,
        "status": "PROPOSED_COMPETING_HYPOTHESIS",
        "experiment_plan": {
            "objective": f"检验 {model_label} 对 {target_label} 的 {horizon_label} 预测是否符合该假设。",
            "design": f"{model_label} 与持久性基线模型在相同数据、切分和指标下进行比较。",
            "treatment_model": model,
            "control_model": "persistence" if key != "persistence" else "学习模型候选组",
            "data_protocol": (
                "按时间顺序划分训练集 / 验证集 / 锁定测试集；预处理仅在训练集拟合；"
                "锁定测试集在设计冻结后仅评估一次。"
            ),
            "window_and_horizon": (
                (
                    f"软测（当前时刻，h=0）"
                    if horizon is None
                    else f"历史窗口 {window_steps} 步，预测时域 {horizon} 步"
                )
                + (
                    f"，采样间隔 {sampling_interval_seconds} 秒。"
                    if sampling_interval_seconds is not None
                    else "。"
                )
            ),
            "model_configuration": configuration,
            "primary_endpoint": primary_metric,
            "secondary_endpoints": secondary_metrics,
            "random_seed": "不适用" if key == "persistence" else random_seed,
            "confirmation_rule": confirmation_rule,
            "falsification_rule": falsification_rule,
            "robustness_checks": [
                "多随机种子复验（Persistence 除外）",
                "多个冻结时间块复验",
                "稳态、升负荷、降负荷和方向变化工况分层",
            ],
            "execution_steps": [
                "冻结数据版本、目标、特征、窗口和预测时域",
                "仅在训练集上拟合预处理器与模型",
                f"在验证集上计算 {primary_metric} 及次要指标",
                "冻结选择后在锁定测试集上一次性评估",
                "保存预测值、指标、参数快照、运行时间和审计结论",
            ],
            "planned_outputs": [
                "验证集指标表",
                "锁定测试集指标表",
                "逐样本预测结果",
                "模型与参数快照",
                "复现信息与实验核验记录",
            ],
            "execution_status": "核心单次比较已执行；稳健性扩展列为下一轮计划。",
        },
    }


class ScientificResearchPlanGenerator:
    """Map existing contracts into a report without new scientific reasoning."""

    @staticmethod
    def _consistency_warnings(
        data: ScientificResearchPlanGeneratorInput,
    ) -> list[str]:
        warnings: list[str] = []
        checks = {
            "problem_id": (
                data.research_problem.problem_id, data.hypothesis.problem_id,
                data.experiment_plan.problem_id, data.experiment_contract.problem_id,
            ),
            "hypothesis_id": (
                data.hypothesis.hypothesis_id, data.experiment_plan.hypothesis_id,
                data.experiment_contract.hypothesis_id,
                data.scientific_result.hypothesis_id,
            ),
            "experiment_id": (
                data.experiment_contract.experiment_id,
                data.experiment_result.experiment_id,
                data.experiment_audit.experiment_id,
                data.scientific_result.experiment_id,
            ),
        }
        for field, values in checks.items():
            if any(value is None or not str(value).strip() for value in values):
                warnings.append(f"consistency_warning:{field}:missing")
            elif len({str(value) for value in values}) != 1:
                warnings.append(f"consistency_warning:{field}:mismatch")
        return warnings

    @staticmethod
    def _model_results(result: ExperimentResult) -> list[ModelResultSnapshot]:
        return [
            ModelResultSnapshot(
                model_name=record.model_name,
                fit_success=record.fit_success,
                fit_converged=record.fit_converged,
                runtime_seconds=record.runtime_seconds,
                model_configuration=dict(record.model_configuration or {}),
                validation_metrics=dict(record.validation_metrics),
                locked_test_metrics=dict(record.locked_test_metrics),
                warnings=tuple(record.warnings),
                failure_reason=record.failure_reason,
                sample_counts=_sample_counts(record),
                random_seed=record.random_seed,
                device=record.device,
                artifact_provenance=dict(record.artifact_provenance),
            )
            for record in result.model_records.values()
        ]

    @staticmethod
    def _references(bundle: EvidenceBundle | None) -> list[ReferenceEntry]:
        if bundle is None:
            return []
        references = []
        for item in bundle.evidence:
            if item.source_type == "DATA_OBSERVATION":
                references.append(ReferenceEntry(
                    evidence_id=item.evidence_id,
                    title=item.title,
                    citation=item.citation,
                    formatted_citation=(
                        f"[数据证据] {item.title}（本地冻结数据集可复现实验，非外部论文）"
                    ),
                    citation_style=None,
                    source_type=item.source_type,
                    source_url=item.source_url,
                    document_id=item.document_id,
                    page_number=item.page_number,
                    chunk_id=item.chunk_id,
                    claim_support=_enum_value(item.claim_support),
                    applicability=_enum_value(item.applicability),
                    citation_verified=False,
                    semantic_verified=False,
                    core_claim_eligible=False,
                    supported_claims=[item.verification_rationale],
                    scope_limits=[
                        "这是实验数据观察证据，不是外部论文；不用于证明因果关系。",
                        "Unity 实际干预结果仍待回传核验。",
                    ],
                ))
                continue
            if not (item.citation_verified and item.semantic_verified):
                continue
            references.append(ReferenceEntry(
                evidence_id=item.evidence_id,
                title=item.title,
                citation=item.citation,
                formatted_citation=item.formatted_citation,
                citation_style=(
                    "GB/T 7714—2015"
                    if item.formatted_citation
                    else None
                ),
                source_type=item.source_type,
                source_url=item.source_url,
                document_id=item.document_id,
                page_number=item.page_number,
                chunk_id=item.chunk_id,
                claim_support=_enum_value(item.claim_support),
                applicability=_enum_value(item.applicability),
                citation_verified=item.citation_verified,
                semantic_verified=item.semantic_verified,
                core_claim_eligible=item.core_claim_eligible,
                supported_claims=[item.verification_rationale],
                scope_limits=([] if item.core_claim_eligible else [
                    "该文献仅作为背景或假设启发，不支撑核心科学结论。"
                ]),
            ))
        return references

    @staticmethod
    def _technical_stack(contract: ExperimentContract) -> list[dict[str, Any]]:
        stack = [
            {"category": "data_preprocessing", "method_name": "pandas/NumPy",
             "purpose": "读取、清洗并按时间顺序组织数据", "required": True},
            {"category": "evaluation", "method_name": "chronological validation + locked test",
             "purpose": "使用验证集选模并隔离锁定测试集", "required": True},
            {"category": "audit", "method_name": "ExperimentAudit",
             "purpose": "检查数据冻结、泄漏、基线和指标有效性", "required": True},
        ]
        sklearn_models = {"ridge", "bayesianridge", "elasticnet", "pls", "svr", "rf", "mlp", "knn", "hgb"}
        torch_models = {"transformer", "lstm", "gru", "dlinear", "patchtst", "itransformer", "timesnet"}
        candidates = set(contract.candidate_models)
        if candidates & sklearn_models:
            stack.append({"category": "machine_learning", "method_name": "scikit-learn",
                          "purpose": "执行经典机器学习候选模型", "required": True})
        if candidates & torch_models:
            stack.append({"category": "deep_learning", "method_name": "PyTorch",
                          "purpose": "执行深度时序候选模型", "required": True})
        if contract.baseline_models or contract.reference_models:
            stack.append({"category": "baseline", "method_name": ", ".join(
                contract.baseline_models or contract.reference_models),
                "purpose": "提供预声明对照基线", "required": True})
        return stack

    @staticmethod
    def _abstract(data: ScientificResearchPlanGeneratorInput) -> str:
        contract, result = data.experiment_contract, data.experiment_result
        validation = {
            name: dict(record.validation_metrics)
            for name, record in result.model_records.items()
        }
        locked = {
            name: dict(record.locked_test_metrics)
            for name, record in result.model_records.items()
        }
        selected = _best_model(validation, contract.primary_metric)
        locked_best = _best_model(locked, contract.primary_metric)
        return (
            f"本研究围绕“{data.research_problem.original_question}”开展建模与验证。"
            f"候选模型在同一批数据上按时间顺序训练和比较，依据验证集 {contract.primary_metric} 选定模型，"
            "锁定测试集仅用于检验结果是否稳定。"
            f"本次选择的模型为 {_display_model(selected)}，锁定测试集表现最佳的模型为 "
            f"{_display_model(locked_best)}；研究结论为“{_display_verdict(data.scientific_result.verdict)}”。"
            "结论适用于本次数据范围和设定工况。"
        )

    def generate(
        self, input_data: ScientificResearchPlanGeneratorInput,
    ) -> ScientificResearchPlan:
        warnings = self._consistency_warnings(input_data)
        problem, hypothesis = input_data.research_problem, input_data.hypothesis
        plan, contract = input_data.experiment_plan, input_data.experiment_contract
        result, audit = input_data.experiment_result, input_data.experiment_audit
        scientific = input_data.scientific_result
        model_results = self._model_results(result)
        validation_by_model = {
            item.model_name: dict(item.validation_metrics or {}) for item in model_results
        }
        locked_by_model = {
            item.model_name: dict(item.locked_test_metrics or {}) for item in model_results
        }
        protocol_selected = _best_model(validation_by_model, contract.primary_metric)
        locked_best = _best_model(locked_by_model, contract.primary_metric)
        baseline_primary = _metric_value(result.baseline_metrics, contract.primary_metric)
        comparison_rows = []
        for item in model_results:
            locked_primary = _metric_value(item.locked_test_metrics, contract.primary_metric)
            improvement = None
            if (
                baseline_primary not in (None, 0.0)
                and locked_primary is not None
                and contract.primary_metric.casefold() != "r2"
            ):
                improvement = (baseline_primary - locked_primary) / abs(baseline_primary) * 100.0
            comparison_rows.append({
                "model": item.model_name,
                "validation_primary": _metric_value(item.validation_metrics, contract.primary_metric),
                "locked_test_primary": locked_primary,
                "locked_test_improvement_vs_baseline_pct": improvement,
                "protocol_selected": item.model_name == protocol_selected,
                "locked_test_best": item.model_name == locked_best,
            })
        metric_unit = (result.normalized_metrics or {}).get("metric_unit")
        unit_warning = (
            contract.target_variable == "steam_volumetric_flow"
            and str(metric_unit or "").casefold() in {"t/h", "tonne/h", "ton/h"}
        )
        artifacts = list(result.artifacts)
        for record in result.model_records.values():
            artifacts.extend(record.artifact_paths)
        proposed_features = list(
            contract.execution_requirements.get("proposed_additional_features", [])
        )

        return ScientificResearchPlan(
            metadata=ScientificResearchPlanMetadata(
                schema_version="boilermind.scientific_hypothesis_research_plan.v2",
                report_id=f"SRP-{contract.experiment_id}",
                generated_at=datetime.now(timezone.utc),
                problem_id=problem.problem_id,
                hypothesis_id=hypothesis.hypothesis_id,
                plan_id=plan.plan_id,
                experiment_id=contract.experiment_id,
                report_status="failed" if warnings else "complete",
                run_id=input_data.run_id,
                report_phase=("POST_ITERATION" if input_data.iteration_occurred else "POST_EXPERIMENT"),
            ),
            paper_title=(
                (
                    f"{contract.prediction_horizon_steps}步时域下"
                    if contract.prediction_horizon_steps is not None
                    else "软测（当前时刻）"
                )
                + f"{_display_target(contract.target_variable)}预测与受约束控制优化研究"
            ),
            paper_abstract=None if warnings else PaperAbstractSection(
                background=problem.original_question,
                objective=problem.objective,
                methods=f"{hypothesis.verification_intent}; {plan.experimental_design}",
                expected_results=hypothesis.expected_observation,
                observed_results=(
                    f"验证集按 {contract.primary_metric} 选择 "
                    f"{_display_model(protocol_selected)}；锁定测试集表现最佳的模型为 "
                    f"{_display_model(locked_best)}。"
                ),
                conclusion=(
                    f"研究结论为“{_display_verdict(scientific.verdict)}”。"
                    "模型选择基于验证集，锁定测试集仅用于独立检验。"
                ),
                limitations="；".join([*hypothesis.evidence_gaps, *audit.issues]) or "限于当前数据集和预声明实验范围。",
                rendered_text=self._abstract(input_data),
            ),
            problem_statement=ProblemStatementSection(
                original_question=problem.original_question,
                research_object=problem.research_object,
                target_variable=problem.target_variable,
                objective=problem.objective,
                metrics=list(problem.metrics),
                target_inference_reason=problem.target_inference_reason,
                operating_condition=problem.operating_condition,
                manipulated_variables=tuple(problem.manipulated_variables),
                observed_variables=tuple(problem.observed_variables),
                context_variables=tuple(problem.context_variables),
                research_goal=problem.research_goal,
                success_criteria=list(problem.success_criteria),
                constraints=list(problem.constraints),
                current_limitation=problem.original_question,
                research_gap=("；".join(hypothesis.evidence_gaps) or "需要通过预声明实验验证当前假设。"),
                limitation_evidence_ids=[
                    item.evidence_id
                    for item in (
                        input_data.evidence_bundle.evidence
                        if input_data.evidence_bundle
                        else []
                    )
                ],
                scope_boundary=[problem.operating_condition, *problem.constraints],
            ),
            rationale=RationaleSection(
                research_significance=hypothesis.research_significance,
                hypothesis_statement=hypothesis.hypothesis,
                mechanism_chain=hypothesis.mechanism_chain,
                mechanism_steps=[step.model_dump(mode="json") for step in hypothesis.mechanism_steps],
                related_variables=list(hypothesis.related_variables),
                applicability_conditions=list(hypothesis.applicability_conditions),
                verification_intent=hypothesis.verification_intent,
                expected_observation=hypothesis.expected_observation,
                assumptions=list(hypothesis.assumptions),
                counter_mechanisms=list(hypothesis.counter_mechanisms),
                evidence_gaps=list(hypothesis.evidence_gaps),
                novelty_axis=hypothesis.novelty_axis,
                evidence_bundle_sha256=hypothesis.evidence_bundle_sha256,
                confirmation_criteria=tuple(hypothesis.confirmation_criteria),
                falsification_criteria=tuple(hypothesis.falsification_criteria),
                innovation_point=hypothesis.novelty_axis,
                reasoning_chain=[{
                    "step_index": index,
                    "premise": hypothesis.hypothesis if index == 1 else hypothesis.mechanism_steps[index - 2].statement,
                    "inference": step.statement,
                    "conclusion": hypothesis.expected_observation if index == len(hypothesis.mechanism_steps) else "进入下一机制步骤",
                    "source_type": "VERIFIED_EVIDENCE" if step.evidence_ids else "DETERMINISTIC_REASONING",
                    "source_ids": list(step.evidence_ids),
                } for index, step in enumerate(hypothesis.mechanism_steps, start=1)],
                competing_hypotheses=[
                    _model_hypothesis(
                        name, contract.target_variable, contract.prediction_horizon_steps,
                        window_steps=contract.window_steps,
                        sampling_interval_seconds=contract.sampling_interval_seconds,
                        primary_metric=contract.primary_metric,
                        secondary_metrics=list(contract.secondary_metrics),
                        random_seed=contract.random_seed,
                    )
                    for name in dict.fromkeys(contract.candidate_models)
                ],
            ),
            technical_details=TechnicalDetailsSection(
                experiment_type=contract.experiment_type or plan.experiment_type,
                required_operations=list(contract.required_operations),
                window_steps=contract.window_steps,
                prediction_horizon_steps=contract.prediction_horizon_steps,
                sampling_interval_seconds=contract.sampling_interval_seconds,
                random_seed=contract.random_seed,
                execution_requirements=dict(contract.execution_requirements),
                allowed_devices=list(contract.allowed_devices),
                reuse_checkpoint_models=list(contract.reuse_checkpoint_models),
                technical_stack=self._technical_stack(contract),
                preprocessing_policy=["预处理参数仅允许在训练集上拟合"],
                leakage_prevention_policy=["验证集用于选择，locked-test 不得参与设计或选模"],
                reproducibility_controls=[f"random_seed={contract.random_seed}", f"dataset_sha256={contract.dataset_hash}"],
            ),
            dataset=DatasetSection(
                source=contract.dataset_id,
                dataset_id=contract.dataset_id,
                dataset_hash=contract.dataset_hash,
                dataset_path=None,
                target=contract.target_variable,
                input_variables=list(contract.input_variables),
                train_split=contract.train_split,
                validation_split=contract.validation_split,
                locked_test_split=contract.test_split,
                scaler_fit_scope=None,
                chronological_split=None,
                sample_counts=_dataset_counts(result),
                historical_data_scope=problem.operating_condition,
                proposed_additional_features=proposed_features,
                collection_required=bool(proposed_features),
                collection_description=(
                    "需要采集实验契约中显式声明的附加特征。"
                    if proposed_features else None
                ),
                train_only_preprocessing=True,
                locked_test_used_for_selection=contract.locked_test_used_for_selection,
            ),
            methods=MethodsSection(
                objective=plan.objective,
                experimental_design=plan.experimental_design,
                baseline_description=plan.baseline_description,
                intervention_description=plan.intervention_description,
                control=dict(contract.control),
                treatment=dict(contract.treatment),
                recommended_models=list(contract.recommended_models),
                executable_models=list(contract.executable_models),
                candidate_models=list(contract.candidate_models),
                reference_models=tuple(contract.reference_models),
                model_selection_rationale=contract.model_selection_rationale,
                model_substitution_reason=contract.model_substitution_reason,
                primary_metric=contract.primary_metric,
                secondary_metrics=tuple(contract.secondary_metrics),
                locked_test_used_for_selection=contract.locked_test_used_for_selection,
                execution_backend=plan.execution_backend,
                allow_partial_failure=contract.allow_partial_failure,
                max_runtime_per_model=contract.max_runtime_per_model,
                max_epochs=contract.max_epochs,
                confirmation_criteria=list(contract.confirmation_criteria),
                falsification_criteria=list(contract.falsification_criteria),
                implementation_steps=[
                    {"step_index": 1, "name": "冻结数据", "description": "校验数据集标识与SHA256"},
                    {"step_index": 2, "name": "时间切分", "description": "建立train/validation/locked-test顺序切分"},
                    {"step_index": 3, "name": "训练与选模", "description": "仅使用train训练并使用validation选择"},
                    {"step_index": 4, "name": "锁定测试", "description": "冻结设计后一次性评估locked-test"},
                    {"step_index": 5, "name": "审计与裁决", "description": "执行ExperimentAudit并应用确认/证伪条件"},
                ],
                hyperparameter_policy="超参数只能依据训练集和验证集确定。",
                model_selection_policy="按预声明验证指标选择，禁止使用locked-test。",
                validation_policy="chronological validation",
                locked_test_policy="locked-test仅用于冻结后的最终评估。",
            ),
            experiments=ExperimentsSection(
                experiment_id=result.experiment_id,
                status=_enum_value(result.status),
                started_at=result.started_at,
                completed_at=result.completed_at,
                model_results=model_results,
                execution_notes=list(result.execution_notes),
                artifacts=list(dict.fromkeys(artifacts)),
                baselines=[{"baseline_id": name, "model_name": name,
                            "rationale": "预声明参考模型", "sample_alignment": "与候选模型使用相同评估样本"}
                           for name in contract.baseline_models],
                metric_definitions=[{"metric_name": name,
                                     "direction": "MAXIMIZE" if name.upper() == "R2" else "MINIMIZE",
                                     "role": "PRIMARY" if name == contract.primary_metric else "SECONDARY",
                                     "evaluation_split": "validation_and_locked_test"}
                                    for name in contract.metrics],
                expected_observation=hypothesis.expected_observation,
            ),
            baselines=list(contract.baseline_models),
            metrics=MetricsSection(
                planned_metrics=list(contract.metrics),
                primary_metric=contract.primary_metric,
                secondary_metrics=list(contract.secondary_metrics),
                validation_metrics_by_model=validation_by_model,
                locked_test_metrics_by_model=locked_by_model,
                baseline_metrics=dict(result.baseline_metrics),
                control_metrics=dict(result.control_metrics),
                treatment_metrics=dict(result.treatment_metrics),
                metric_deltas=dict(result.metric_deltas),
                metric_unit=metric_unit,
            ),
            results=ResultsSection(
                overall_metrics=dict(result.metrics),
                baseline_metrics=dict(result.baseline_metrics),
                candidate_locked_test_metrics={name: dict(values) for name, values in result.candidate_locked_test_metrics.items()},
                control_metrics=dict(result.control_metrics),
                treatment_metrics=dict(result.treatment_metrics),
                metric_deltas=dict(result.metric_deltas),
                result_status=_enum_value(result.status).upper(),
                feasibility_basis="ACTUAL_EXECUTION",
                achieved_criteria=list(scientific.achieved_criteria),
                failed_criteria=list(scientific.failed_criteria),
                scientific_verdict=_enum_value(scientific.verdict),
                verdict_rationale=scientific.rationale,
                experiment_valid=result.experiment_valid,
                audit_issues=list(audit.issues),
                protocol_selected_model=protocol_selected,
                locked_test_best_model=locked_best,
                selection_interpretation=(
                    f"依据验证集结果，本次选定 {_display_model(protocol_selected)}；"
                    f"锁定测试集表现最佳的模型为 {_display_model(locked_best)}。"
                    "若两者不一致，应通过更多时间段复验后再判断稳定性。"
                ),
                model_comparison_rows=comparison_rows,
            ),
            scientific_verdict=ScientificVerdictSnapshot(
                verdict=scientific.verdict,
                rationale=scientific.rationale,
                achieved_criteria=tuple(scientific.achieved_criteria),
                failed_criteria=tuple(scientific.failed_criteria),
                source_hypothesis_id=scientific.hypothesis_id,
                source_experiment_id=scientific.experiment_id,
            ),
            experiment_validity=ExperimentValiditySnapshot(
                experiment_valid=result.experiment_valid,
                execution_valid=audit.execution_valid,
                dataset_frozen=audit.dataset_frozen,
                leakage_check_passed=audit.leakage_check_passed,
                baseline_valid=audit.baseline_valid,
                metric_check_passed=audit.metric_check_passed,
                issues=tuple([*result.experiment_validity_issues, *audit.issues]),
                validity_source="ExperimentAudit",
            ),
            references=self._references(input_data.evidence_bundle),
            limitations=list(dict.fromkeys([
                *hypothesis.evidence_gaps,
                *result.experiment_validity_issues,
                *audit.issues,
                *warnings,
                *(["指标单位与体积流量目标不一致：产物声明为 t/h，需复核单位规范化逻辑。"] if unit_warning else []),
                *(
                    ["外部文献证据不可用（联网文献未启用或检索为空），系统未补造文献；报告证据仅来自本地实验数据观察。"]
                    if not input_data.evidence_bundle
                    or all(
                        item.source_type == "DATA_OBSERVATION"
                        for item in input_data.evidence_bundle.evidence
                    )
                    else []
                ),
            ])),
            provenance=[
                ProvenanceEntry(source_object="ResearchProblemSpec", source_id=problem.problem_id, schema_version=None),
                ProvenanceEntry(
                    source_object="EvidenceBundle",
                    source_id=(
                        input_data.evidence_bundle.bundle_id
                        if input_data.evidence_bundle
                        else None
                    ),
                    schema_version=None,
                ),
                ProvenanceEntry(source_object="ScientificHypothesis", source_id=hypothesis.hypothesis_id, schema_version=None),
                ProvenanceEntry(source_object="ExperimentPlan", source_id=plan.plan_id, schema_version=None),
                ProvenanceEntry(source_object="ExperimentContract", source_id=contract.experiment_id, schema_version=None),
                ProvenanceEntry(source_object="ExperimentResult", source_id=result.experiment_id, schema_version=None),
                ProvenanceEntry(source_object="ExperimentAudit", source_id=audit.experiment_id, schema_version=None),
                ProvenanceEntry(source_object="ScientificResult", source_id=scientific.experiment_id, schema_version=None),
            ],
            research_trace=[ResearchTraceEntry.model_validate(item) for item in input_data.research_trace],
            final_selection=FinalPlanSelection(
                hypothesis_id=hypothesis.hypothesis_id,
                round_index=input_data.round_index,
                revision_index=input_data.revision_index,
                plan_id=plan.plan_id,
                experiment_id=contract.experiment_id,
                selection_reason=input_data.selection_reason,
                iteration_occurred=input_data.iteration_occurred,
                fallback_applied=input_data.fallback_applied,
            ),
        )
