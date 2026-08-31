"""PDF rendering for the final 《科研假设与研究计划》.

仅使用冻结后的报告字段；章节按赛题（XH-202619）要求的标准化字段组织：
待研究问题 / 解决思路 / 必要的技术手段 / 数据集（Source+Target）/ 标题 /
摘要 / 方法论 / 实验设计（Baselines+Metrics）/ 实验结果 / 参考论文。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from boilermind.core.contracts.scientific_research_plan import ScientificResearchPlan

_FONT_CANDIDATES = (
    ("MSYH", r"C:\Windows\Fonts\msyh.ttc", 0),
    ("SimHei", r"C:\Windows\Fonts\simhei.ttf", None),
    ("SimSun", r"C:\Windows\Fonts\simsun.ttc", 0),
)

_REGISTERED = False
_FONT_NAME = "Helvetica"


def _ensure_fonts() -> str:
    global _REGISTERED, _FONT_NAME
    if _REGISTERED:
        return _FONT_NAME
    for name, path, subfont_index in _FONT_CANDIDATES:
        candidate = Path(path)
        if not candidate.is_file():
            continue
        try:
            if subfont_index is None:
                pdfmetrics.registerFont(TTFont(name, str(candidate)))
            else:
                pdfmetrics.registerFont(
                    TTFont(name, str(candidate), subfontIndex=subfont_index)
                )
            _FONT_NAME = name
            _REGISTERED = True
            return name
        except Exception:
            continue
    _REGISTERED = True
    return "Helvetica"


def _text(value: Any) -> str:
    if value is None or value == "":
        return "未提供"
    if isinstance(value, (dict, list, tuple)):
        import json

        value = json.dumps(value, ensure_ascii=False)
    return _localize_report_text(str(value))


def _localize_report_text(text: str) -> str:
    if text.casefold() == "supported":
        return "得到支持"
    labels = {
        "all_candidates_worse_than_reference_on:MAE": "所有候选模型在 MAE 上均劣于参考模型",
        "any_candidate_better_than_reference_on:MAE": "任一候选模型在 MAE 上优于参考模型",
        "locked_test_not_used_for_selection": "锁定测试集不参与模型选择",
        "locked_test": "锁定测试集",
        "validation_only_model_selection": "仅使用验证集进行模型选择",
        "reference_model_comparison": "参考基线对比",
        "chronological_validation": "按时间顺序验证",
        "locked_test_evaluation": "锁定测试集评估",
        "model_comparison": "模型对比",
        "UNIFIED_EXPERIMENT_EXECUTION": "统一实验执行",
        "target_profile:": "目标变量：",
        "compare_prediction_accuracy": "比较预测精度",
        "unspecified": "未特别限定",
        "falsified": "未获支持",
        "partially_supported": "部分支持",
        "insufficient_evidence": "证据不足",
        "chronological validation + locked test": "按时间顺序验证 + 锁定测试集",
        "train/validation/locked-test": "训练集/验证集/锁定测试集",
        "ExperimentAudit": "实验审计",
        "ResearchProblemSpec": "研究问题规范",
        "EvidenceBundle": "证据集合",
        "ScientificHypothesis": "科学假设",
        "ExperimentPlan": "实验方案",
        "ExperimentContract": "实验契约",
        "ExperimentResult": "实验结果",
        "ScientificResult": "科学结果",
        "FIRST_ROUND_NO_ITERATION": "首轮无需迭代",
        "COMPLETED": "已完成",
        "data_preprocessing": "数据预处理",
        "evaluation": "评估",
        "audit": "审计",
        "baseline": "基线",
        "current_operating_point": "当前工况点",
        "PROPOSED_COMPETING_HYPOTHESIS": "待验证",
        "Persistence": "持久性基线模型",
        "ACHIEVED_RISE_PCT": "目标提升比例（%）",
        "PRESSURE_MAX_MPA": "最大压力（MPa）",
        "hgb_control_optimizer": "HGB 控制优化器",
        "hgb_soft_sensor_validation": "HGB 软测验证",
        "constrained_control_optimization": "受约束控制优化",
        "drum_pressure <= 23 MPa": "汽包压力不超过 23 MPa",
        "each control adjustment within ±25%": "各控制变量的单次调整幅度不超过 ±25%",
        "constraint_search_from_validation_operating_point": "基于验证集工况点的约束搜索",
        "constraint_search": "约束搜索",
        "unity_result_export": "Unity 结果导出",
        "steam_volumetric_flow": "蒸汽体积流量 V",
        "POST_EXPERIMENT": "实验后",
        "POST_ITERATION": "迭代后",
        "ACTUAL_EXECUTION": "实际执行结果",
        "validation_and_locked_test": "验证集与锁定测试集",
        "validation_only_model_selection": "仅使用验证集进行模型选择",
        "chronological validation": "按时间顺序验证",
        "chronological locked test": "按时间顺序锁定测试",
        "chronological": "按时间顺序",
        "test_frozen": "锁定测试集",
        "locked test 最优": "锁定测试集最优",
        "使用 locked test 回选": "使用锁定测试集回选",
        "locked test 不参与": "锁定测试集不参与",
        "locked-test": "锁定测试集",
        "locked test": "锁定测试集",
        "validation-only": "仅验证集",
        "validation": "验证集",
        "train": "训练集",
        "VERIFIED": "已核验",
        "True": "是",
        "False": "否",
        "None": "未提供",
    }
    for source, target in labels.items():
        text = text.replace(source, target)
    text = re.sub(r"\bsupported\b", "得到支持", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfalsified\b", "未获支持", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgru\b", "GRU", text, flags=re.IGNORECASE)
    text = re.sub(r"\blstm\b", "LSTM", text, flags=re.IGNORECASE)
    return re.sub(r"\bcompleted\b", "已完成", text, flags=re.IGNORECASE)


def _number(value: Any) -> str:
    try:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "—"


def _metric(metrics: dict[str, Any] | None, name: str) -> Any:
    wanted = str(name or "").casefold()
    for key, value in (metrics or {}).items():
        if str(key).casefold() == wanted:
            return value
    return None


class ScientificResearchPlanPdfRenderer:
    """Deterministic PDF rendering; never infers new scientific claims."""

    def render(self, plan: ScientificResearchPlan, target: str | Path) -> Path:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        font = _ensure_fonts()
        styles = self._styles(font)
        story: list[Any] = []
        self._title_page(story, styles, plan)
        self._abstract_section(story, styles, plan)
        self._problem_section(story, styles, plan)
        self._rationale_section(story, styles, plan)
        self._hypotheses_section(story, styles, plan)
        self._plan_section(story, styles, plan)
        self._results_section(story, styles, plan)
        self._verdict_section(story, styles, plan)
        self._references_section(story, styles, plan)
        self._limitations_section(story, styles, plan)
        document = SimpleDocTemplate(
            str(path), pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=18 * mm,
            title=_text(plan.paper_title) or "科研假设与研究计划",
            author="BoilerMind 研究工作台",
        )
        document.build(story)
        return path

    def _styles(self, font: str) -> dict[str, ParagraphStyle]:
        return {
            "title": ParagraphStyle(
                "title", fontName=font, fontSize=17, leading=22,
                alignment=TA_CENTER, spaceAfter=6,
            ),
            "subtitle": ParagraphStyle(
                "subtitle", fontName=font, fontSize=10.5, leading=15,
                alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                spaceAfter=4,
            ),
            "h1": ParagraphStyle(
                "h1", fontName=font, fontSize=14, leading=19,
                spaceBefore=14, spaceAfter=7, textColor=colors.HexColor("#0f3d6e"),
            ),
            "h2": ParagraphStyle(
                "h2", fontName=font, fontSize=12, leading=16,
                spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1a1a1a"),
            ),
            "body": ParagraphStyle(
                "body", fontName=font, fontSize=10, leading=15,
                spaceAfter=4, wordWrap="CJK",
            ),
            "bullet": ParagraphStyle(
                "bullet", fontName=font, fontSize=9.5, leading=14,
                leftIndent=12, spaceAfter=2, wordWrap="CJK",
            ),
            "meta": ParagraphStyle(
                "meta", fontName=font, fontSize=9, leading=13,
                alignment=TA_CENTER, textColor=colors.HexColor("#666666"),
            ),
        }

    def _title_page(self, story, styles, plan: ScientificResearchPlan) -> None:
        metadata = plan.metadata
        story.append(Spacer(1, 28 * mm))
        story.append(Paragraph("《科研假设与研究计划》", styles["title"]))
        story.append(Paragraph(_text(plan.paper_title), styles["subtitle"]))
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph(f"生成时间：{_text(metadata.generated_at if metadata else None)}", styles["meta"]))
        story.append(Paragraph(
            "模型选择依据验证集；锁定测试集仅用于独立检验。",
            styles["meta"],
        ))
        story.append(PageBreak())

    def _abstract_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        abstract = plan.paper_abstract
        if abstract is None:
            return
        story.append(Paragraph("1. 研究摘要", styles["h1"]))
        summary = (
            getattr(abstract, "rendered_text", None)
            or getattr(abstract, "polished_text", None)
            or _text(abstract)
        )
        story.append(Paragraph(_text(summary), styles["body"]))
        details = [
            ("背景", abstract.background),
            ("研究目标", abstract.objective),
            ("方法", abstract.methods),
            ("预期结果", abstract.expected_results),
            ("观察结果", abstract.observed_results),
            ("结论", abstract.conclusion),
            ("局限", abstract.limitations),
        ]
        for label, value in details:
            if value:
                story.append(Paragraph(f"<b>{label}：</b>{_text(value)}", styles["bullet"]))
        story.append(Spacer(1, 2 * mm))

    def _problem_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        problem = plan.problem_statement
        if problem is None:
            return
        story.append(Paragraph("2. 待研究问题", styles["h1"]))
        rows = [
            ("原始问题", problem.original_question),
            ("研究对象", problem.research_object),
            ("目标变量", problem.target_variable),
            ("研究目标", problem.research_goal),
            ("评价指标", "、".join(problem.metrics or [])),
            ("工况范围", problem.operating_condition),
            ("研究缺口", problem.research_gap),
            ("边界约束", "；".join(problem.scope_boundary or [])),
        ]
        self._kv_table(story, styles, rows)

    def _rationale_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        rationale = plan.rationale
        if rationale is None:
            return
        story.append(Paragraph("3. 解决思路", styles["h1"]))
        story.append(Paragraph("3.1 创新点", styles["h2"]))
        story.append(Paragraph(_text(rationale.innovation_point or rationale.research_significance), styles["body"]))
        story.append(Paragraph("3.2 推理链条", styles["h2"]))
        story.append(Paragraph(_text(rationale.mechanism_chain or "未提供"), styles["body"]))
        if rationale.mechanism_steps:
            story.append(Paragraph("3.3 分步机理", styles["h2"]))
            for step in rationale.mechanism_steps:
                if isinstance(step, dict):
                    story.append(Paragraph(
                        f"步骤 {_text(step.get('step'))}：{_text(step.get('statement'))}",
                        styles["bullet"],
                    ))

    def _hypotheses_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        rationale = plan.rationale
        if rationale is None:
            return
        story.append(Paragraph("4. 科研假设", styles["h1"]))
        story.append(Paragraph("4.1 主假设", styles["h2"]))
        story.append(Paragraph(_text(rationale.hypothesis_statement), styles["body"]))
        rows = [
            ("验证意图", rationale.verification_intent),
            ("预期观察", rationale.expected_observation),
            ("适用条件", "；".join(rationale.applicability_conditions or [])),
            ("证据缺口", "；".join(rationale.evidence_gaps or [])),
        ]
        self._kv_table(story, styles, rows)
        story.append(Paragraph("4.2 确认与证伪标准", styles["h2"]))
        story.append(Paragraph(
            "确认标准：" + ("；".join(rationale.confirmation_criteria or []) or "无"),
            styles["bullet"],
        ))
        story.append(Paragraph(
            "证伪标准：" + ("；".join(rationale.falsification_criteria or []) or "无"),
            styles["bullet"],
        ))
        if rationale.assumptions:
            story.append(Paragraph("关键假设条件：" + "；".join(rationale.assumptions), styles["bullet"]))
        competing = rationale.competing_hypotheses or []
        if competing:
            story.append(Paragraph("4.3 竞争子假设", styles["h2"]))
            for index, item in enumerate(competing, start=1):
                if not isinstance(item, dict):
                    continue
                story.append(Paragraph(
                    f"H{index} · {_text(item.get('title') or item.get('model'))}："
                    f"{_text(item.get('statement'))}",
                    styles["bullet"],
                ))
                if item.get("expected_observation"):
                    story.append(Paragraph(
                        f"预期观察：{_text(item.get('expected_observation'))}", styles["bullet"],
                    ))
                plan_item = item.get("experiment_plan") or {}
                if plan_item:
                    story.append(Paragraph("实验规划书：", styles["bullet"]))
                    for label, key in (
                        ("实验目的", "objective"), ("实验设计", "design"),
                        ("主指标", "primary_endpoint"), ("确认规则", "confirmation_rule"),
                        ("证伪规则", "falsification_rule"),
                    ):
                        value = plan_item.get(key)
                        if value:
                            story.append(Paragraph(f"· {label}：{_text(value)}", styles["bullet"]))

    def _plan_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        story.append(Paragraph("5. 科研规划书", styles["h1"]))
        technical, dataset = plan.technical_details, plan.dataset
        methods, experiments = plan.methods, plan.experiments
        story.append(Paragraph("5.1 必要的技术手段", styles["h2"]))
        if technical:
            rows = [
                ("实验类型", technical.experiment_type),
                ("必要操作", "；".join(technical.required_operations or [])),
                ("历史窗口", technical.window_steps),
                ("预测时域", technical.prediction_horizon_steps),
                ("采样间隔", technical.sampling_interval_seconds),
                ("随机种子", technical.random_seed),
            ]
            self._kv_table(story, styles, rows)
            stack = technical.technical_stack or []
            if stack:
                story.append(Paragraph("技术栈：", styles["body"]))
                data = [["类别", "方法", "用途"]]
                for item in stack:
                    data.append([
                        _text(item.get("category")),
                        _text(item.get("method_name")),
                        _text(item.get("purpose")),
                    ])
                self._generic_table(story, styles, data, widths=[28 * mm, 48 * mm, 88 * mm])
        story.append(Paragraph("5.2 数据集", styles["h2"]))
        if dataset:
            rows = [
                ("历史推演数据", dataset.dataset_id),
                ("目标变量", dataset.target),
                ("输入特征", f"{len(dataset.input_variables or [])} 个"),
                ("训练/验证/锁定测试", (
                    f"{_text(dataset.train_split)} / {_text(dataset.validation_split)} / "
                    f"{_text(dataset.locked_test_split)}"
                )),
                ("样本数", _text(dataset.sample_counts)),
                ("数据合规状态", dataset.source_compliance_status),
            ]
            self._kv_table(story, styles, rows)
            if dataset.proposed_additional_features or dataset.collection_required:
                story.append(Paragraph("拟采集数据特征：", styles["body"]))
                if dataset.proposed_additional_features:
                    story.append(Paragraph(
                        "；".join(_text(item) for item in dataset.proposed_additional_features), styles["bullet"],
                    ))
                story.append(Paragraph(
                    f"需补充采集：{'是' if dataset.collection_required else '否'}"
                    + (f"；{_text(dataset.collection_description)}" if dataset.collection_description else ""),
                    styles["bullet"],
                ))
            else:
                story.append(Paragraph(
                    "拟采集数据特征：本实验基于现有冻结数据完成，无需补充采集新特征。",
                    styles["bullet"],
                ))
        story.append(Paragraph("5.3 方法论", styles["h2"]))
        if methods:
            rows = [
                ("研究目标", methods.objective),
                ("实验设计", methods.experimental_design),
                ("候选模型", "、".join(methods.candidate_models or [])),
                ("参考模型", "、".join(methods.reference_models or [])),
                ("主指标", methods.primary_metric),
                ("次要指标", "、".join(methods.secondary_metrics or [])),
            ]
            self._kv_table(story, styles, rows)
            story.append(Paragraph("实施步骤：", styles["body"]))
            for item in methods.implementation_steps:
                story.append(Paragraph(
                    f"{_text(item.get('step_index'))}. {_text(item.get('name'))}："
                    f"{_text(item.get('description'))}",
                    styles["bullet"],
                ))
        story.append(Paragraph("5.4 实验设计（基线与指标）", styles["h2"]))
        if plan.baselines:
            story.append(Paragraph("基线对比：" + "、".join(_text(item) for item in plan.baselines), styles["bullet"]))
        if plan.metrics:
            story.append(Paragraph(
                "评估指标：" + "、".join(
                    [
                        *([_text(plan.metrics.primary_metric)] if plan.metrics.primary_metric else []),
                        *(_text(item) for item in (plan.metrics.secondary_metrics or [])),
                    ]
                ),
                styles["bullet"],
            ))
        if experiments and experiments.model_results:
            data = [["模型", "拟合成功", "运行时(秒)"]]
            for item in experiments.model_results:
                data.append([
                    _text(item.model_name),
                    "是" if item.fit_success else "否",
                    _number(item.runtime_seconds),
                ])
            self._generic_table(story, styles, data, widths=[60 * mm, 40 * mm, 50 * mm])

    def _results_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        results, metrics = plan.results, plan.metrics
        if results is None and metrics is None:
            return
        story.append(Paragraph("6. 实验结果", styles["h1"]))
        if results:
            story.append(Paragraph(
                f"执行状态：{_text(results.result_status)}；验证基础：{_text(results.feasibility_basis)}",
                styles["body"],
            ))
        rows = results.model_comparison_rows if results else []
        if rows:
            story.append(Paragraph("模型比较（按主指标）：", styles["body"]))
            data = [["模型", "验证集", "锁定测试集", "相对基线改善(%)"]]
            for row in rows:
                data.append([
                    _text(row.get("model")),
                    _number(row.get("validation_primary")),
                    _number(row.get("locked_test_primary")),
                    _number(row.get("locked_test_improvement_vs_baseline_pct")),
                ])
            self._generic_table(story, styles, data, widths=[40 * mm, 38 * mm, 42 * mm, 44 * mm])
        if metrics:
            validation = metrics.validation_metrics_by_model or {}
            locked = metrics.locked_test_metrics_by_model or {}
            models = list(dict.fromkeys([*validation, *locked]))
            if models:
                story.append(Paragraph("各模型指标明细：", styles["body"]))
                for model in models:
                    story.append(Paragraph(f"· {_text(model)}", styles["bullet"]))
                    self._kv_table(story, styles, [
                        ("验证集", _text(validation.get(model))),
                        ("锁定测试集", _text(locked.get(model))),
                    ])
            if metrics.baseline_metrics:
                story.append(Paragraph("基线指标：", styles["body"]))
                story.append(Paragraph(_text(metrics.baseline_metrics), styles["bullet"]))
        if results:
            for label, values in (
                ("达成标准", results.achieved_criteria),
                ("未达成标准", results.failed_criteria),
            ):
                if values:
                    story.append(Paragraph(f"{label}：{'；'.join(_text(item) for item in values)}", styles["bullet"]))

    def _verdict_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        verdict, validity = plan.scientific_verdict, plan.experiment_validity
        story.append(Paragraph("7. 研究结论与验证情况", styles["h1"]))
        if verdict:
            story.append(Paragraph(
                f"研究结论：{_text(verdict.verdict)}", styles["body"],
            ))
            story.append(Paragraph(f"裁决依据：{_text(verdict.rationale)}", styles["bullet"]))
        if validity:
            story.append(Paragraph(
                f"实验有效性：{'有效' if validity.experiment_valid else '无效'}（"
                f"执行有效性 {'有效' if validity.execution_valid else '无效'}，"
                f"数据冻结 {'是' if validity.dataset_frozen else '否'}，"
                f"防泄漏检查 {'通过' if validity.leakage_check_passed else '未通过'}）",
                styles["body"],
            ))
            if validity.issues:
                story.append(Paragraph("需要关注：" + "；".join(_text(item) for item in validity.issues), styles["bullet"]))

    def _references_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        references = plan.references or []
        story.append(Paragraph("8. 参考论文", styles["h1"]))
        if not references:
            story.append(Paragraph("未检索到可绑定引用；系统不虚构文献，报告证据仅来自本地实验数据。", styles["body"]))
            return
        for index, item in enumerate(references, start=1):
            story.append(Paragraph(
                f"[{index}] {_text(item.formatted_citation or item.citation or item.title)}",
                styles["bullet"],
            ))

    def _limitations_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        limitations = plan.limitations or []
        story.append(Paragraph("9. 局限性与适用范围", styles["h1"]))
        if not limitations:
            story.append(Paragraph("无显式局限性记录。", styles["body"]))
            return
        for item in limitations:
            story.append(Paragraph(f"· {_text(item)}", styles["bullet"]))

    def _appendix_section(self, story, styles, plan: ScientificResearchPlan) -> None:
        story.append(Paragraph("10. 审计附录", styles["h1"]))
        trace = plan.research_trace or []
        if trace:
            story.append(Paragraph("研究轨迹：", styles["body"]))
            for entry in trace:
                story.append(Paragraph(
                    f"· {_text(entry.plan_id)} / {_text(entry.experiment_id)}："
                    f"状态 {_text(entry.status)}；目标达成 {'是' if entry.target_met else '否'}",
                    styles["bullet"],
                ))
        selection = plan.final_selection
        if selection:
            story.append(Paragraph(
                f"最终选择：假设 {_text(selection.hypothesis_id)}，第 {_text(selection.round_index)} 轮 "
                f"第 {_text(selection.revision_index)} 次修订（{_text(selection.selection_reason)}）",
                styles["bullet"],
            ))
        provenance = plan.provenance or []
        if provenance:
            story.append(Paragraph(
                "来源对象：" + "；".join(
                    f"{_text(item.source_object)}#{_text(item.source_id)}" for item in provenance
                ),
                styles["bullet"],
            ))
        story.append(Paragraph(
            "本报告所有数值与裁决均来自冻结实验产物；LLM 仅用于语言表述润色，不改变科学结论。",
            styles["meta"],
        ))

    def _kv_table(self, story, styles, rows) -> None:
        data = [[Paragraph(_text(label), styles["body"]), Paragraph(_text(value), styles["body"])] for label, value in rows]
        self._generic_table(story, styles, data, widths=[48 * mm, 116 * mm])

    def _generic_table(self, story, styles, data, widths) -> None:
        if len(data) <= 1:
            return
        localized_data = [
            [cell if isinstance(cell, Paragraph) else _text(cell) for cell in row]
            for row in data
        ]
        table = Table(localized_data, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f9")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d4e0")),
            ("FONTNAME", (0, 0), (-1, -1), _FONT_NAME),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 12),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
        story.append(Spacer(1, 3 * mm))
