from __future__ import annotations

import json
from typing import Any, Callable

from boilermind.core.llm_client import LLMClient


class NarrativeReportValidationError(ValueError):
    pass


def _canonical_metrics(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    aliases = {
        "MAE": ("MAE", "mae_m3_s", "mae_t_h", "mae"),
        "RMSE": ("RMSE", "rmse_m3_s", "rmse_t_h", "rmse"),
        "R2": ("R2", "r2"),
        "MBE": ("MBE", "mbe_m3_s", "mbe_t_h", "mbe"),
    }
    canonical = {}
    for name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in metrics:
                canonical[name] = metrics[candidate]
                break
    return canonical or None


def build_structured_research_record(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "boilermind.structured_research_record.v2",
        "run_id": state["run_id"],
        "question": state["question"],
        "research_problem": state.get("research_problem"),
        "hypotheses": state.get("hypotheses", []),
        "ranking_snapshots": state.get("ranking_snapshots", []),
        "batches": state.get("batches", []),
        "evidence_bundle": state.get("evidence_bundle"),
        "experiment_memory_bundle": state.get("experiment_memory_bundle"),
    }


def build_narrative_prompt_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep report facts auditable without sending the full runtime object."""
    hypotheses = []
    for item in record.get("hypotheses", []):
        hypotheses.append({
            key: item.get(key)
            for key in (
                "hypothesis_id", "title", "hypothesis_statement",
                "expected_observation", "falsification_condition",
                "generation_source", "scientific_design",
            )
            if item.get(key) not in (None, "", [], {})
        })

    batches = []
    for batch in record.get("batches", []):
        members = []
        for member in batch.get("members", []):
            contract = member.get("contract") or {}
            outcome = member.get("outcome") or {}
            result = outcome.get("experiment_result") or {}
            model_metrics = {}
            for model, model_record in (result.get("model_records") or {}).items():
                model_metrics[str(model)] = {
                    "validation_metrics": _canonical_metrics(
                        model_record.get("validation_metrics")
                    ),
                    "locked_test_metrics": _canonical_metrics(
                        model_record.get("locked_test_metrics")
                    ),
                    "fit_success": model_record.get("fit_success"),
                    "runtime_seconds": model_record.get("runtime_seconds"),
                }
            members.append({
                "hypothesis_id": member.get("hypothesis_id"),
                "experiment_id": contract.get("experiment_id"),
                "status": member.get("status"),
                "candidate_models": contract.get("candidate_models"),
                "reference_models": contract.get("reference_models"),
                "target": contract.get("target"),
                "prediction_horizon_steps": contract.get("prediction_horizon_steps"),
                "primary_metric": contract.get("primary_metric"),
                "baseline_metrics": _canonical_metrics(
                    result.get("baseline_metrics")
                ),
                "model_metrics": model_metrics,
                "audit": outcome.get("audit"),
                "scientific_result": outcome.get("scientific_result"),
                "closure_ok": outcome.get("closure_ok"),
            })
        batches.append({
            "batch_id": batch.get("batch_id"),
            "status": batch.get("status"),
            "members": members,
        })

    return {
        "schema_version": "boilermind.narrative_prompt_record.v1",
        "run_id": record.get("run_id"),
        "question": record.get("question"),
        "research_problem": record.get("research_problem"),
        "hypotheses": hypotheses,
        "batches": batches,
    }


def _deterministic_narrative(record: dict[str, Any]) -> str:
    """Deterministic narrative fallback that only restates frozen facts."""
    problem = record.get("research_problem") or {}
    lines = [
        "# BoilerMind 科研报告",
        "",
        f"Run ID：{record.get('run_id', '')}",
        f"研究问题：{record.get('question', '')}",
        "",
        "## 研究问题",
        "",
        f"- 研究对象：{problem.get('research_object', '未提供')}",
        f"- 目标变量：{problem.get('target_variable', '未提供')}",
        f"- 研究目标：{problem.get('research_goal', '未提供')}",
        f"- 工况范围：{problem.get('operating_condition', '未提供')}",
        "",
        "## 科研假设",
        "",
    ]
    for hypothesis in record.get("hypotheses", []):
        lines.append(
            f"- {hypothesis.get('hypothesis_id', '')}："
            f"{hypothesis.get('hypothesis_statement') or hypothesis.get('hypothesis', '')}"
        )
    lines.extend(["", "## 实验执行与科学裁决", ""])
    for batch in record.get("batches", []):
        for member in batch.get("members", []):
            contract = member.get("contract") or {}
            outcome = member.get("outcome") or {}
            result = outcome.get("experiment_result") or {}
            scientific = outcome.get("scientific_result") or {}
            metrics = result.get("metrics") or {}
            metric_text = "；".join(
                f"{name}={value}" for name, value in metrics.items()
            ) or "未记录"
            lines.append(
                f"- 实验 {contract.get('experiment_id', '')}（假设 "
                f"{member.get('hypothesis_id', '')}）：状态 "
                f"{member.get('status', '')}；指标：{metric_text}；"
                f"裁决：{scientific.get('verdict', '未记录')}。"
            )
    lines.extend([
        "",
        "## 证据说明",
        "",
        "本报告由确定性回退生成：仅复述冻结实验事实，不新增科学推理；"
        "外部文献不可用时系统未补造文献，证据以本地实验数据观察为准。",
        "",
        "冻结标识：" + "；".join(
            str(item) for item in [
                record.get("run_id"),
                *[
                    member.get("hypothesis_id")
                    for batch in record.get("batches", [])
                    for member in batch.get("members", [])
                ],
                *[
                    (member.get("contract") or {}).get("experiment_id")
                    for batch in record.get("batches", [])
                    for member in batch.get("members", [])
                ],
            ] if item
        ),
        "",
    ])
    return "\n".join(lines)


def write_narrative_report(
    record: dict[str, Any],
    *,
    generate: Callable[[str], str] | None = None,
) -> str:
    prompt_record = build_narrative_prompt_record(record)
    prompt = """
你是BoilerMind科研报告撰写器。只能总结所给结构化事实，不得修改实验ID、
假设ID、模型、指标数值、审计状态或科学裁决，不得新增未绑定引用，不得把
INSUFFICIENT_EVIDENCE升级为支持，也不得把可观察前提升级为机理结论。
输出中文科研报告正文。

STRUCTURED_RECORD:
""" + json.dumps(prompt_record, ensure_ascii=False, sort_keys=True)
    if generate is not None:
        return str(generate(prompt)).strip()
    # 默认走 LLM；任何不可用（未配置/无密钥/网络失败）都回退到确定性总结，
    # 保证控制链路在离线环境下仍能生成完整叙事报告。
    try:
        return str(LLMClient().generate(prompt)).strip()
    except Exception:
        return _deterministic_narrative(record)


def bind_frozen_identifiers(record: dict[str, Any], narrative: str) -> str:
    """Deterministically bind run/experiment identity without editing prose facts."""
    identifiers = [str(record.get("run_id", "")).strip()]
    for batch in record.get("batches", []):
        for member in batch.get("members", []):
            identifiers.append(str(member.get("hypothesis_id", "")).strip())
            contract = member.get("contract") or {}
            identifiers.append(str(contract.get("experiment_id", "")).strip())
    identifiers = list(dict.fromkeys(item for item in identifiers if item))
    if all(item in narrative for item in identifiers):
        return narrative
    header = "# BoilerMind 科研报告\n\n冻结标识：" + "；".join(identifiers)
    return header + "\n\n" + narrative.strip()


def validate_narrative_report(record: dict[str, Any], narrative: str) -> None:
    if not narrative:
        raise NarrativeReportValidationError("empty_narrative_report")
    required_tokens: set[str] = {str(record.get("run_id", ""))}
    for batch in record.get("batches", []):
        for member in batch.get("members", []):
            required_tokens.add(str(member.get("hypothesis_id", "")))
            contract = member.get("contract") or {}
            required_tokens.add(str(contract.get("experiment_id", "")))
    missing = sorted(token for token in required_tokens if token and token not in narrative)
    if missing:
        raise NarrativeReportValidationError(
            "narrative_missing_frozen_identifiers:" + ",".join(missing)
        )
