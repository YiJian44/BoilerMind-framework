"""Standalone service interface for generating all scientific-plan artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from pydantic import Field

from boilermind.core.llm_client import LLMClient
from boilermind.core.contracts.base import ContractModel
from .pipeline_report_adapter import PipelineReportAdapter
from .final_plan_selector import FinalResearchPlanSelector
from .scientific_research_plan_generator import (
    ScientificResearchPlanGenerator, ScientificResearchPlanGeneratorInput,
)
from .scientific_research_plan_renderer import ScientificResearchPlanRenderer
from .scientific_research_plan_pdf import ScientificResearchPlanPdfRenderer


class ScientificResearchPlanResponse(ContractModel):
    schema_version: str = "boilermind.scientific_research_plan.response.v1"
    run_id: str
    status: str
    report: dict | None = None
    json_path: str | None = None
    markdown_path: str | None = None
    word_path: str | None = None
    pdf_path: str | None = None
    manifest_path: str | None = None
    sha256: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class ScientificResearchPlanService:
    """Backend-compatible facade; accepts strict input or full-pipeline context."""

    def __init__(self) -> None:
        self.adapter = PipelineReportAdapter()
        self.generator = ScientificResearchPlanGenerator()
        self.renderer = ScientificResearchPlanRenderer()
        self.pdf_renderer = ScientificResearchPlanPdfRenderer()
        self.selector = FinalResearchPlanSelector()

    def _polish_abstract(
        self, plan,
    ) -> None:
        """LLM 仅润色语言；任何不可用（未配置/网络失败/非法输出）都保持确定性文本。"""
        abstract = getattr(plan, "paper_abstract", None)
        if abstract is None or not hasattr(abstract, "rendered_text"):
            return
        try:
            facts = {
                "paper_title": plan.paper_title,
                "rendered_text": abstract.rendered_text,
                "background": abstract.background,
                "objective": abstract.objective,
                "methods": abstract.methods,
                "expected_results": abstract.expected_results,
                "observed_results": abstract.observed_results,
                "conclusion": abstract.conclusion,
            }
            prompt = (
                "你是 BoilerMind 科研报告的语言润色器。只优化中文表达与行文，"
                "严禁改动任何实验 ID、假设 ID、模型名、指标数值、科学裁决、数据集哈希等事实，"
                "不得新增引用，不得把 INSUFFICIENT_EVIDENCE 升级为支持，"
                "不得把可观察前提升级为机理结论。基于下方结构化事实，"
                "输出不超过 300 字的执行摘要（一段，不写标题，不使用 Markdown 列表）。\n"
                "STRUCTURED_FACTS:\n"
                + json.dumps(facts, ensure_ascii=False, sort_keys=True)
                + '\n只输出 JSON：{"polished_text": "..."}'
            )
            raw = str(LLMClient().generate(prompt)).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            payload = json.loads(match.group(0)) if match else {}
            polished = str(payload.get("polished_text") or "").strip()
            if polished and len(polished) >= 20:
                plan.paper_abstract = abstract.model_copy(update={"polished_text": polished})
        except Exception:
            return

    def generate_from_run_state(
        self, state: dict, *, output_dir: str | Path,
    ) -> ScientificResearchPlanResponse:
        """Adapter used by ResearchOrchestrator after its experiment loop."""
        run_id = str(state.get("run_id", ""))
        try:
            selected = self.selector.select(state)
            outcome = selected.member.get("outcome") or {}
            trace = []
            for batch in state.get("batches", []):
                for member in batch.get("members", []):
                    member_outcome = member.get("outcome") or {}
                    result = member_outcome.get("experiment_result") or {}
                    contract = member.get("contract") or {}
                    scientific = member_outcome.get("scientific_result") or {}
                    if result:
                        trace.append({
                            "plan_id": contract.get("plan_id"),
                            "experiment_id": contract.get("experiment_id"),
                            "status": member.get("status"),
                            "metrics": result.get("metrics", {}),
                            "target_met": str(scientific.get("verdict", "")).upper() == "SUPPORTED",
                            "reason": scientific.get("rationale"),
                        })
            context = {
                "research_problem": state.get("research_problem"),
                "evidence_bundle": state.get("evidence_bundle"),
                "hypotheses": state.get("hypotheses", []),
                "selected_hypothesis_id": selected.selection.hypothesis_id,
                "experiment_plan": selected.member.get("plan"),
                "experiment_contract": selected.member.get("contract"),
                "experiment_result": outcome.get("experiment_result"),
                "experiment_audit": outcome.get("audit"),
                "scientific_result": outcome.get("scientific_result"),
                "research_trace": trace,
                "validity_source": "ExperimentAudit",
                "experiment_valid": bool((outcome.get("audit") or {}).get("execution_valid")),
            }
            data = self.adapter.adapt(context).model_copy(update={
                "run_id": run_id,
                "round_index": selected.selection.round_index,
                "revision_index": selected.selection.revision_index,
                "selection_reason": selected.selection.selection_reason,
                "iteration_occurred": selected.selection.iteration_occurred,
                "fallback_applied": selected.selection.fallback_applied,
            })
            return self.generate(run_id=run_id, output_dir=output_dir, input_data=data)
        except Exception as exc:
            return ScientificResearchPlanResponse(
                run_id=run_id, status="FAILED",
                errors=[f"{type(exc).__name__}:{exc}"],
            )

    def generate(
        self, *, run_id: str, output_dir: str | Path,
        input_data: ScientificResearchPlanGeneratorInput | None = None,
        pipeline_context: dict | None = None,
    ) -> ScientificResearchPlanResponse:
        try:
            data = input_data or self.adapter.adapt(pipeline_context or {})
            if data.run_id is None:
                data = data.model_copy(update={"run_id": run_id})
            plan = self.generator.generate(data)
            self._polish_abstract(plan)
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            json_path = root / "scientific_research_plan.json"
            markdown_path = root / "scientific_research_plan.md"
            word_path = root / "scientific_research_plan.docx"
            pdf_path = root / "scientific_research_plan.pdf"
            manifest_path = root / "manifest.json"
            _atomic_text(json_path, plan.model_dump_json(indent=2))
            _atomic_text(markdown_path, self.renderer.render_markdown(plan))
            word_tmp = word_path.with_suffix(".docx.tmp")
            self.renderer.render_docx(plan, word_tmp)
            os.replace(word_tmp, word_path)
            self.pdf_renderer.render(plan, pdf_path)
            hashes = {
                "json": _sha256(json_path),
                "markdown": _sha256(markdown_path),
                "word": _sha256(word_path),
                "pdf": _sha256(pdf_path),
            }
            manifest = {
                "schema_version": "boilermind.scientific_research_plan.manifest.v1",
                "run_id": run_id,
                "selection": plan.final_selection.model_dump(mode="json") if plan.final_selection else None,
                "artifacts": {
                    "json": str(json_path), "markdown": str(markdown_path),
                    "word": str(word_path), "pdf": str(pdf_path),
                },
                "sha256": hashes,
            }
            _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
            return ScientificResearchPlanResponse(
                run_id=run_id,
                status="GENERATED" if plan.metadata and plan.metadata.report_status == "complete" else "GENERATED_WITH_LIMITATIONS",
                report=plan.model_dump(mode="json"),
                json_path=str(json_path), markdown_path=str(markdown_path),
                word_path=str(word_path), pdf_path=str(pdf_path),
                manifest_path=str(manifest_path), sha256=hashes,
            )
        except Exception as exc:
            return ScientificResearchPlanResponse(
                run_id=run_id, status="FAILED",
                errors=[f"{type(exc).__name__}:{exc}"],
            )
