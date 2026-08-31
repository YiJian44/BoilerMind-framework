"""Stable frontend projection for BoilerMind research run state."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


STAGES = (
    ("problem", "问题拆解"),
    ("evidence", "证据与假设"),
    ("plan", "实验方案"),
    ("execution", "实验执行"),
    ("evaluation", "科学评价"),
    ("report", "科研报告"),
)


_STAGE_SUMMARIES = {
    "problem": "已解析研究目标、目标变量、运行工况与时间范围，形成冻结科研问题。",
    "evidence": "已检索候选文献证据并生成候选科研假设；未执行语义核验时仅展示本地库统计。",
    "plan": "已生成冻结实验方案：候选模型、参考基线、主指标与执行门禁。",
    "execution": "已按冻结契约执行真实模型实验，并记录 Validation 与 Locked-test 指标。",
    "evaluation": "已按预声明协议比较候选模型，形成科学裁决与工程门禁判断。",
    "report": "已生成《科研假设与研究计划》报告及可下载产物。",
}


# 展示层缓存：同一 run 的降级文献候选只计算一次，避免每次轮询重建 BM25 索引。
_LOCAL_RAG_SOURCE: Any | None = None
_DEGRADED_CANDIDATES_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _read_unity_state(run_id: str, run_dir: Path) -> dict:
    # run_dir = <run_root>/<run_id>，状态文件与后端一致地放在 <run_root 上级>/control
    global_state = run_dir.parents[1] / "control" / "unity_push_state.json"
    if not global_state.is_file():
        return {}
    try:
        payload = json.loads(global_state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return (payload.get("runs") or {}).get(run_id) or {}


def _completed_member(state: dict[str, Any]) -> dict[str, Any] | None:
    for batch in reversed(state.get("batches") or []):
        for member in batch.get("members") or []:
            if member.get("status") == "COMPLETED" and member.get("outcome"):
                return member
    return None


def _stage_facts(state: dict[str, Any], run_dir: Path) -> list[bool]:
    member = _completed_member(state)
    outcome = (member or {}).get("outcome") or {}
    report_ready = bool(
        (state.get("report") or {}).get("structured_path")
        or (run_dir / "structured_report.json").is_file()
        or (run_dir / "scientific_research_plan" / "scientific_research_plan.json").is_file()
    )
    return [
        bool(state.get("research_problem")),
        bool(state.get("hypotheses"))
        and (bool(state.get("evidence_bundle")) or _literature_degraded(state)),
        any((batch.get("members") or []) for batch in state.get("batches") or []),
        bool(member),
        bool(outcome.get("scientific_result") or outcome.get("audit")),
        report_ready,
    ]


def _literature_degraded(state: dict[str, Any]) -> bool:
    return any(
        trace.get("stage") == "literature_retrieval"
        and trace.get("status") == "FAILED"
        for trace in (state.get("stage_traces") or [])
    )


def _stages(state: dict[str, Any], run_dir: Path) -> tuple[list[dict[str, Any]], str]:
    facts = _stage_facts(state, run_dir)
    terminal = state.get("status") in {
        "COMPLETED", "COMPLETED_WITH_REPORT_WARNING", "FAILED",
        "NO_EXECUTABLE_HYPOTHESES",
    }
    first_incomplete = next((index for index, done in enumerate(facts) if not done), 5)
    current = 5 if all(facts) else first_incomplete
    rows = []
    traces = state.get("stage_traces") or []
    for index, (stage_id, name) in enumerate(STAGES):
        if facts[index]:
            status = "completed"
        elif state.get("status") == "FAILED" and index == current:
            status = "failed"
        elif not terminal and index == current:
            status = "running"
        else:
            status = "waiting"
        if status == "completed":
            summary = _STAGE_SUMMARIES[stage_id]
        elif status == "running":
            summary = f"正在执行：{_STAGE_SUMMARIES[stage_id]}"
        elif status == "failed":
            summary = "该阶段执行失败，失败原因见后端运行日志与错误提示。"
        else:
            summary = "等待前序阶段完成后自动展开。"
        rows.append({
            "stage_id": stage_id,
            "name": name,
            "status": status,
            "summary": summary,
            "progress_percent": 100 if status == "completed" else 25 if status == "running" else 0,
            "trace_count": len(traces) if stage_id in {"problem", "evidence"} else None,
        })
    return rows, STAGES[current][0]


def _hypotheses(state: dict[str, Any]) -> list[dict[str, Any]]:
    plans = {}
    for batch in state.get("batches") or []:
        for member in batch.get("members") or []:
            plans[member.get("hypothesis_id")] = member.get("plan") or {}
    latest_rank = (state.get("ranking_snapshots") or [{}])[-1]
    entries = latest_rank.get("entries") or []
    ranks = {
        item.get("hypothesis_id"): index
        for index, item in enumerate(entries, start=1)
    }
    scores = {
        item.get("hypothesis_id"): item.get("dynamic_score")
        for item in entries
        if item.get("hypothesis_id")
    }
    result = []
    for item in state.get("hypotheses") or []:
        hypothesis_id = item.get("hypothesis_id") or item.get("id")
        hypothesis_state = (state.get("hypothesis_states") or {}).get(hypothesis_id) or {}
        result.append({
            "hypothesis_id": hypothesis_id,
            "title": item.get("title"),
            "statement": item.get("hypothesis") or item.get("hypothesis_statement"),
            "mechanism_chain": item.get("mechanism_chain") or item.get("engineering_mechanism") or item.get("mechanism"),
            "expected_observation": item.get("expected_observation") or item.get("inference"),
            "confirmation_criteria": item.get("confirmation_criteria") or [],
            "falsification_criteria": item.get("falsification_criteria") or [],
            "evidence_ids": item.get("evidence_ids") or [],
            "rank": ranks.get(hypothesis_id),
            "status": hypothesis_state.get("latest_verdict") or item.get("status"),
            "experiment_plan": plans.get(hypothesis_id),
            "selection_reason": "",
        })
    top = next((entry for entry in result if entry["rank"] == 1), None)
    if top is not None and top.get("hypothesis_id"):
        score = scores.get(top["hypothesis_id"])
        top["selection_reason"] = (
            "按预声明综合评分（历史支持/问题相关性/可复现性/可证伪性）排序最高"
            + (f"，dynamic_score={score:.4f}" if isinstance(score, (int, float)) else "")
            + "。"
        )
    return result


def _model_selection_summary(
    *,
    rows: list[dict[str, Any]],
    primary: str,
    selected_model: str | None,
    locked_best_model: str | None,
    validation_candidates: dict[str, float],
    locked_candidates: dict[str, float],
    contract: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Additive model-selection rationale + detail for the frontend.

    Pure projection of persisted experiment data; never mutates state.
    """
    lower_is_better = primary.casefold() != "r2"
    persistence_locked = locked_candidates.get("persistence")
    detail_rows = []
    for row in rows:
        model = row.get("model")
        validation = _metric(row.get("validation") or {}, primary)
        locked = _metric(row.get("locked_test") or {}, primary)
        improvement = None
        if (
            model != "persistence"
            and locked is not None
            and persistence_locked is not None
            and persistence_locked != 0
        ):
            improvement = (
                (persistence_locked - locked) / abs(persistence_locked) * 100.0
                if lower_is_better
                else locked - persistence_locked
            )
        detail_rows.append({
            "model": model,
            "fit_success": bool(row.get("fit_success")),
            "failure_reason": row.get("failure_reason"),
            "validation_primary": validation,
            "locked_test_primary": locked,
            "improvement_vs_persistence": (
                round(improvement, 4) if improvement is not None else None
            ),
        })
    if not validation_candidates:
        rationale = "候选模型均未完成 Validation 评估，当前无可用的模型选择依据。"
    else:
        lines = []
        selected_value = (
            validation_candidates.get(selected_model) if selected_model else None
        )
        if selected_value is not None:
            lines.append(
                f"按冻结协议，以 {primary} 为选择指标，正式选择 {selected_model}"
                f"（Validation {primary} = {selected_value:.4f}）。"
            )
        else:
            lines.append(f"按冻结协议，正式选择 {selected_model or '—'}。")
        locked_value = (
            locked_candidates.get(locked_best_model) if locked_best_model else None
        )
        if locked_value is not None:
            lines.append(
                f"独立 Locked-test 上最优为 {locked_best_model}"
                f"（{primary} = {locked_value:.4f}）。"
            )
        if selected_model and locked_best_model:
            if selected_model == locked_best_model:
                lines.append("正式选择与 Locked-test 最优一致，结论协议一致性良好。")
            else:
                lines.append("二者不一致，不得用 Locked-test 回选模型，建议追加跨时间块复验。")
        selected_improvement = next(
            (
                row.get("improvement_vs_persistence")
                for row in detail_rows
                if row["model"] == selected_model
            ),
            None,
        )
        if selected_improvement is not None:
            lines.append(
                f"相对 Persistence，{primary} 改善 {selected_improvement:.2f}%。"
                if lower_is_better
                else f"相对 Persistence，R2 提升 {selected_improvement:.4f}。"
            )
        rationale = " ".join(lines)
    plan_rationale = str(contract.get("model_selection_rationale") or "").strip()
    if plan_rationale:
        rationale = f"{plan_rationale} {rationale}" if rationale else plan_rationale
    detail = {
        "primary_metric": primary,
        "protocol_selected_model": selected_model,
        "locked_test_best_model": locked_best_model,
        "protocol_consistent_with_locked": (
            bool(selected_model and selected_model == locked_best_model)
            if selected_model and locked_best_model
            else None
        ),
        "execution_valid": bool((audit or {}).get("execution_valid")),
        "rows": detail_rows,
    }
    return detail, rationale


def _ranking_trace(state: dict[str, Any]) -> dict[str, Any] | None:
    """Additive projection of the per-round hypothesis ranking trace.

    Feeds the frontend re-ranking replay (排行榜回放) and the iteration
    feedback summary. Never mutates state.
    """
    snapshots = state.get("ranking_snapshots") or []
    if not snapshots:
        return None
    rounds = []
    for snapshot in snapshots:
        entries = []
        for item in snapshot.get("entries") or []:
            entries.append({
                "hypothesis_id": item.get("hypothesis_id"),
                "historical_support": item.get("historical_support"),
                "prior_score": item.get("prior_score"),
                "cumulative_feedback": item.get("cumulative_feedback"),
                "dynamic_score": item.get("dynamic_score"),
                "eligible": bool(item.get("eligible")),
                "dropped_reasons": item.get("dropped_reasons") or [],
            })
        rounds.append({
            "round_index": snapshot.get("round_index"),
            "snapshot_id": snapshot.get("snapshot_id"),
            "entries": entries,
        })
    feedback = []
    for batch in state.get("batches") or []:
        round_index = batch.get("round_index")
        for member in batch.get("members") or []:
            if member.get("status") != "COMPLETED":
                continue
            outcome = member.get("outcome") or {}
            scientific = outcome.get("scientific_result") or {}
            audit = outcome.get("audit") or {}
            feedback.append({
                "round_index": round_index,
                "hypothesis_id": member.get("hypothesis_id"),
                "experiment_id": (outcome.get("experiment_result") or {}).get("experiment_id"),
                "verdict": scientific.get("verdict"),
                "execution_valid": bool((audit or {}).get("execution_valid")),
            })
    return {"rounds": rounds, "feedback": feedback}


def _execution(state: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    member = _completed_member(state)
    if not member:
        return None
    outcome = member.get("outcome") or {}
    result = outcome.get("experiment_result") or {}
    audit = outcome.get("audit") or {}
    scientific = outcome.get("scientific_result") or {}
    records = result.get("model_records") or {}
    experiment_id = result.get("experiment_id")
    rows = []
    for model, record in records.items():
        rows.append({
            "model": model,
            "fit_success": record.get("fit_success"),
            "fit_converged": record.get("fit_converged"),
            "runtime_seconds": record.get("runtime_seconds"),
            "validation": record.get("validation_metrics") or {},
            "locked_test": record.get("locked_test_metrics") or {},
            "sample_counts": {
                "train": record.get("train_samples"),
                "validation": record.get("validation_samples"),
                "test": record.get("test_samples"),
            },
            "random_seed": record.get("random_seed"),
            "device": record.get("device"),
            "epochs_completed": record.get("epochs_completed"),
            "warnings": record.get("warnings") or [],
            "failure_reason": record.get("failure_reason"),
        })
    baseline = result.get("baseline_metrics") or {}
    if baseline:
        rows.append({
            "model": "persistence", "fit_success": True,
            "fit_converged": True, "runtime_seconds": 0.0,
            "validation": {}, "locked_test": baseline,
            "sample_counts": {}, "random_seed": None, "device": None,
            "epochs_completed": None,
            "warnings": [], "failure_reason": None,
        })
    primary = ((member.get("contract") or {}).get("primary_metric") or "MAE")
    validation_candidates = {
        row["model"]: _metric(row["validation"], primary)
        for row in rows if row["model"] != "persistence"
    }
    validation_candidates = {key: value for key, value in validation_candidates.items() if value is not None}
    locked_candidates = {row["model"]: _metric(row["locked_test"], primary) for row in rows}
    locked_candidates = {key: value for key, value in locked_candidates.items() if value is not None}
    chooser = max if primary.casefold() == "r2" else min
    selected_model = (
        chooser(validation_candidates, key=validation_candidates.get)
        if validation_candidates else None
    )
    locked_best_model = (
        chooser(locked_candidates, key=locked_candidates.get)
        if locked_candidates else None
    )
    selection_detail, selection_rationale = _model_selection_summary(
        rows=rows,
        primary=primary,
        selected_model=selected_model,
        locked_best_model=locked_best_model,
        validation_candidates=validation_candidates,
        locked_candidates=locked_candidates,
        contract=(member.get("contract") or {}),
        audit=audit,
    )
    return {
        "experiment_id": experiment_id,
        "primary_metric": primary,
        "secondary_metrics": (member.get("contract") or {}).get("secondary_metrics") or [],
        "metric_unit": result.get("metric_unit"),
        "protocol_selected_model": selected_model,
        "locked_test_best_model": locked_best_model,
        "locked_test_used_for_selection": bool((member.get("contract") or {}).get("locked_test_used_for_selection")),
        "executed_step_ids": [
            str(trace.get("stage"))
            for trace in (state.get("stage_traces") or [])
            if trace.get("status") == "COMPLETED"
        ],
        "environment": _environment_snapshot(run_dir, experiment_id),
        "rows": rows,
        "scientific_result": scientific,
        "audit": audit,
        "selection_rationale": selection_rationale,
        "selection_detail": selection_detail,
    }


def _control(state: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    """Control-optimization-specific frontend card data."""
    member = _completed_member(state)
    if not member:
        return None
    plan = member.get("plan") or {}
    experiment_type = str(plan.get("experiment_type") or "").casefold()
    if "control" not in experiment_type and not plan.get("control"):
        return None
    outcome = member.get("outcome") or {}
    result = outcome.get("experiment_result") or {}
    scientific = outcome.get("scientific_result") or {}
    summary = outcome.get("control_summary") or {}
    control = plan.get("control") or {}
    treatment = plan.get("treatment") or {}
    current_values = control.get("current_values") or {}
    current_volume = control.get("current_volume")
    recommended_values = treatment.get("recommended_values") or {}
    adjustment_ranges = treatment.get("adjustment_ranges") or {}
    metrics = result.get("metrics") or {}
    unity = _read_unity_state(str(state.get("run_id") or ""), run_dir)
    payload_path = summary.get("unity_payload_path") or unity.get("payload_path")
    payload_exists = bool(payload_path and Path(payload_path).is_file())
    push_status = str(unity.get("push_status") or ("payload_generated" if payload_exists else "none"))
    second_verdict = unity.get("second_verdict")
    conclusion_scope = str(
        result.get("conclusion_scope") or "small_model_control_validation"
    )
    return {
        "experiment_type": plan.get("experiment_type"),
        "current": {
            "values": current_values,
            "volume": current_volume,
        },
        "recommended": {
            "ranges": adjustment_ranges,
            "values": recommended_values,
        },
        "constraints": plan.get("hard_constraints") or [],
        "results": {
            "validation_mae": metrics.get("MAE") or summary.get("validation_mae"),
            "current_volume": summary.get("current_volume"),
            "target_volume": summary.get("target_volume"),
            "predicted_volume": summary.get("predicted_volume"),
            "predicted_rise": summary.get("predicted_rise"),
            "feasible_candidates": summary.get("feasible_candidates"),
            "pressure_max_mpa": metrics.get("PRESSURE_MAX_MPA"),
        },
        "conclusion": {
            "verdict": scientific.get("verdict"),
            "rationale": scientific.get("rationale"),
            "scope": conclusion_scope,
            "unity_verified": second_verdict in {"supported", "partially_supported", "falsified"},
            "second_verdict": second_verdict,
        },
        "unity": {
            "status": push_status,
            "payload_generated": bool(payload_exists),
            "payload_path": payload_path,
            "pushed_at": unity.get("pushed_at"),
            "received_at": unity.get("received_at"),
            "executed_at": unity.get("executed_at"),
            "returned_at": unity.get("returned_at"),
            "actual_volume": unity.get("actual_volume"),
            "actual_rise_pct": unity.get("actual_rise_pct"),
            "predicted_volume": summary.get("predicted_volume"),
            "deviation_pct": unity.get("deviation_pct"),
            "notes": unity.get("notes") or [],
        },
    }


def _metric(values: dict[str, Any], name: str) -> float | None:
    wanted = name.casefold()
    for key, value in values.items():
        if str(key).casefold() == wanted:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _environment_snapshot(run_dir: Path, experiment_id: str | None) -> dict[str, Any]:
    """运行环境快照：优先读取实验 manifest 的真实运行记录，缺失时取当前运行时。"""
    if experiment_id:
        manifest = (
            run_dir.parents[1]
            / "experiment_runs"
            / experiment_id
            / "manifest.json"
        )
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                recorded_env = payload.get("environment") or {}
                return {
                    "os": recorded_env.get("os"),
                    "python_version": payload.get("python_version"),
                    "dependency_versions": payload.get("dependency_versions") or {},
                    "adapter": "unified_runner",
                    "metadata": recorded_env.get("metadata"),
                }
            except (OSError, ValueError):
                pass
    try:
        from boilermind.models.execution_environment import ExecutionEnvironment
        import importlib.metadata as importlib_metadata

        env = ExecutionEnvironment.detect().to_dict()
        dependencies: dict[str, str | None] = {}
        for name in ("numpy", "pandas", "scikit-learn", "torch"):
            try:
                dependencies[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                dependencies[name] = None
        return {
            "os": env.get("os"),
            "python_version": env.get("python_version"),
            "dependency_versions": dependencies,
            "adapter": "projection_detect",
            "metadata": env.get("metadata"),
        }
    except Exception:
        return {
            "os": None,
            "python_version": None,
            "dependency_versions": {},
            "adapter": "unknown",
            "metadata": {},
        }


def _local_library_stats() -> dict[str, Any] | None:
    """本地文献库规模统计（展示用，不做任何科学判定）。"""
    rag_root = Path(__file__).resolve().parents[2] / "resources" / "local_rag"
    papers_path = rag_root / "metadata" / "papers.jsonl"
    chunks_path = rag_root / "artifacts" / "chunks" / "chunks.jsonl"
    if not papers_path.is_file() or not chunks_path.is_file():
        return None
    try:
        paper_count = 0
        by_level: Counter[str] = Counter()
        with papers_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                paper_count += 1
                by_level[str(item.get("corpus_level") or "unknown")] += 1
        chunk_count = 0
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk_count += 1
        return {
            "paper_count": paper_count,
            "chunk_count": chunk_count,
            "by_corpus_level": dict(by_level),
        }
    except (OSError, ValueError):
        return None


def _degraded_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    """降级时尽力从本地文献库检索候选摘要，仅用于展示，不替代语义核验。"""
    global _LOCAL_RAG_SOURCE
    run_id = str(state.get("run_id") or "")
    question = str(state.get("question") or "")
    cache_key = (run_id, question)
    if cache_key in _DEGRADED_CANDIDATES_CACHE:
        return _DEGRADED_CANDIDATES_CACHE[cache_key]
    _DEGRADED_CANDIDATES_CACHE[cache_key] = []
    problem_payload = state.get("research_problem")
    try:
        if not isinstance(problem_payload, dict) or not problem_payload.get("problem_id"):
            return []
        from boilermind.core.contracts import ResearchProblemSpec
        from boilermind.evidence.sources.local_rag import LocalRAGSource

        if _LOCAL_RAG_SOURCE is None:
            _LOCAL_RAG_SOURCE = LocalRAGSource(top_k=5)
        problem = ResearchProblemSpec.model_validate(problem_payload)
        candidates = _LOCAL_RAG_SOURCE.retrieve(problem)
        items = [
            {
                "title": candidate.title,
                "citation": candidate.citation or candidate.formatted_citation,
                "corpus_level": candidate.corpus_level,
                "snippet": str(candidate.text or "")[:160],
                "source_type": candidate.source_type,
            }
            for candidate in candidates[:5]
        ]
        _DEGRADED_CANDIDATES_CACHE[cache_key] = items
        return items
    except Exception:
        return []


_ARTIFACT_DISPLAY_NAMES = {
    "structured_report": "结构化研究报告.json",
    "narrative_report": "研究叙述报告.md",
    "scientific_plan_json": "科研假设与研究计划.json",
    "scientific_plan_markdown": "科研假设与研究计划.md",
    "scientific_plan_word": "科研假设与研究计划.docx",
    "scientific_plan_pdf": "科研假设与研究计划.pdf",
    "scientific_plan_manifest": "科研假设与研究计划清单.json",
}


def artifact_display_name(artifact_id: str, fallback: str) -> str:
    return _ARTIFACT_DISPLAY_NAMES.get(artifact_id, fallback)


def _artifacts(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        ("structured_report", run_dir / "structured_report.json"),
        ("narrative_report", run_dir / "narrative_report.md"),
        ("scientific_plan_json", run_dir / "scientific_research_plan" / "scientific_research_plan.json"),
        ("scientific_plan_markdown", run_dir / "scientific_research_plan" / "scientific_research_plan.md"),
        ("scientific_plan_word", run_dir / "scientific_research_plan" / "scientific_research_plan.docx"),
        ("scientific_plan_pdf", run_dir / "scientific_research_plan" / "scientific_research_plan.pdf"),
        ("scientific_plan_manifest", run_dir / "scientific_research_plan" / "manifest.json"),
    ]
    result = []
    for artifact_id, path in candidates:
        if path.is_file():
            result.append({
                "artifact_id": artifact_id,
                "name": artifact_display_name(artifact_id, path.name),
                "mime_type": {
                    ".json": "application/json", ".md": "text/markdown",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".pdf": "application/pdf",
                }.get(path.suffix.casefold(), "application/octet-stream"),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "download_url": None,
            })
    return result


def project_run(state: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Project internal state into the stable frontend schema."""

    stages, current_stage = _stages(state, run_dir)
    status_map = {
        "QUEUED": "queued", "RUNNING": "running", "COMPLETED": "completed",
        "COMPLETED_WITH_REPORT_WARNING": "completed_with_warning",
        "NO_EXECUTABLE_HYPOTHESES": "needs_human_review", "FAILED": "failed",
    }
    artifacts = _artifacts(run_dir)
    run_id = str(state.get("run_id") or "")
    for item in artifacts:
        item["download_url"] = f"/api/v1/research-runs/{run_id}/artifacts/{item['artifact_id']}/download"
    evidence = (state.get("evidence_bundle") or {}).get("evidence") or []
    literature_degraded = _literature_degraded(state)
    revision_source = json.dumps(
        [state.get("status"), state.get("stage_traces"), state.get("batches"), state.get("report")],
        ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8")
    revision = int(hashlib.sha256(revision_source).hexdigest()[:12], 16)
    completed = sum(stage["status"] == "completed" for stage in stages)
    return {
        "schema_version": "boilermind.frontend.research_run.v1",
        "run": {
            "run_id": run_id,
            "question": state.get("question"),
            "status": status_map.get(state.get("status"), "running"),
            "current_stage": current_stage,
            "progress_percent": round(completed / len(STAGES) * 100),
            "started_at": state.get("started_at"),
            "completed_at": state.get("completed_at"),
            "revision": revision,
        },
        "stages": stages,
        "problem": state.get("research_problem"),
        "evidence_summary": {
            "count": len(evidence),
            "items": evidence,
            "degraded": literature_degraded and not evidence,
            "local_stats": _local_library_stats(),
            "degraded_candidates": _degraded_candidates(state) if literature_degraded and not evidence else [],
            "degraded_note": (
                "外部文献语义核验不可用（Qwen/网络未连通），系统未补造文献；"
                "本地文献库可检索但未执行语义核验，"
                "证据以本地实验数据观察为准，报告已在局限性中注明。"
                if literature_degraded and not evidence
                else None
            ),
        },
        "hypotheses": _hypotheses(state),
        "execution": _execution(state, run_dir),
        "control": _control(state, run_dir),
        "report": {"ready": any(item["artifact_id"].startswith("scientific_plan") for item in artifacts)},
        "artifacts": artifacts,
        "ranking": _ranking_trace(state),
        "errors": state.get("errors") or [],
    }
