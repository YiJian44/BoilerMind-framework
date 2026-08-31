from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boilermind.core.contracts import (
    EvidenceTier,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
)


_EXPERIMENT_HEADING = re.compile(r"^###\s+实验\s*(\d+)[:：]\s*(.+?)\s*$")
_FIELD = re.compile(r"^-\s*\*\*[①②③④⑤](?:-⑤)?\s*([^*]+)\*\*(?:（[^）]*）)?[:：]\s*(.*)$")
_PLANNED = re.compile(r"^-\s*\*\*(H[\w_]+)\*\*\s*(.*)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ints(pattern: str, text: str) -> list[int]:
    match = re.search(pattern, text, re.I)
    return [int(value) for value in re.findall(r"\d+", match.group(1))] if match else []


def _models(text: str) -> list[str]:
    known = (
        "persistence", "last-reading", "ridge", "bayesianridge", "hgb", "rf",
        "randomforest", "lstm", "gru", "dlinear", "transformer", "patchtst",
        "itransformer", "timesnet", "tcn", "mtgnn", "csdi", "pls", "elasticnet",
        "residual_lstm",
    )
    lowered = text.lower().replace(" ", "")
    return sorted({model for model in known if model.replace("-", "") in lowered.replace("-", "")})


def _scope(context: str, protocol: str) -> ExperimentScopeSignature:
    text = f"{context} {protocol}"
    lowered = text.lower()
    horizon = _ints(r"(?:horizon\s*[=:]?|\bh\s*[=:])\s*(\d+)", lowered)
    if not horizon:
        horizon = _ints(r"@\s*(\d+)", lowered)
    window = _ints(r"(?:window|窗口)\s*[=:]?\s*(\d+)", lowered)
    feature_count = _ints(r"(\d+)\s*(?:特征|维)", text)
    split = re.search(r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", text)
    split_ratios = [int(v) / 100 for v in split.groups()] if split else []
    if "direct-v" in lowered or "v direct" in lowered or "直接软测量" in text or re.search(r"目标\s*=\s*\*{0,2}v\b", lowered):
        prediction_mode = "direct_volume"
    elif "m@" in lowered or "质量流量" in text or "mass 模式" in text:
        prediction_mode = "mass"
    elif "间接" in text:
        prediction_mode = "indirect_volume"
    else:
        prediction_mode = None
    if "if97" in lowered:
        thermodynamic = "IF97"
    elif "理想气体" in text:
        thermodynamic = "IDEAL_GAS"
    else:
        thermodynamic = None
    target = "steam_volumetric_flow" if prediction_mode in {"direct_volume", "indirect_volume"} else (
        "main_steam_mass_flow" if prediction_mode == "mass" else None
    )
    return ExperimentScopeSignature(
        target_variable=target,
        target_definition=("direct-V" if prediction_mode == "direct_volume" else ("M" if prediction_mode == "mass" else None)),
        target_unit=("m3/s" if target == "steam_volumetric_flow" else ("t/h" if target == "main_steam_mass_flow" else None)),
        prediction_mode=prediction_mode,
        thermodynamic_standard=thermodynamic,
        feature_count=feature_count[0] if feature_count else None,
        window_steps=window[0] if window else None,
        prediction_horizon_steps=horizon[0] if horizon else None,
        sampling_interval_seconds=15 if "15s" in lowered or "15秒" in text else None,
        split_policy="chronological" if "chronological" in lowered or "时间序" in text else None,
        split_ratios=split_ratios,
        regime_definition="ramp_down" if "ramp_down" in lowered else None,
        metrics=sorted({metric.upper() for metric in re.findall(r"\b(MAE|RMSE|R2|MAPE|MBE)\b", text, re.I)}),
        baselines=[model for model in _models(text) if model in {"persistence", "last-reading", "ridge", "bayesianridge"}],
        protocol_status=("SEALED" if "sealed" in lowered else ("MISSING" if "协议" in text and "缺失" in text else "EXPLORATORY")),
    )


def _tier(number: int, protocol: str, result: str, limitations: str) -> EvidenceTier:
    combined = f"{protocol} {result} {limitations}".lower()
    if "未执行" in combined or "未做" in combined:
        return EvidenceTier.PLANNED_NOT_EXECUTED
    if any(word in combined for word in ("执行失败", "框架级", "结果与判定：缺失")):
        return EvidenceTier.ENGINEERING_FAILURE
    if "非 sealed" in combined or "non-sealed" in combined:
        return EvidenceTier.LEGACY_INFORMATIVE
    if number in {9, 10, 11, 12, 15} and "sealed" in combined:
        return EvidenceTier.AUDITED_CONFIRMATORY
    if "sealed" in combined or "决策性结论" in combined:
        return EvidenceTier.AUDITED_EXPLORATORY
    if any(word in combined for word in ("非 sealed", "探索性", "非正式", "协议：缺失")):
        return EvidenceTier.LEGACY_INFORMATIVE
    return EvidenceTier.LEGACY_INFORMATIVE


def _verdict(result: str) -> str | None:
    lowered = result.lower()
    if any(word in lowered for word in ("证伪", "不支持", "被拒", "❌", "fail")):
        if any(word in lowered for word in ("唯一过线", "局部", "但", "overall")):
            return "PARTIALLY_SUPPORTED"
        return "FALSIFIED"
    if any(word in lowered for word in ("支持", "✅", "pass", "成立")):
        return "SUPPORTED"
    return "INFORMATIVE" if result else None


def _record_from_section(
    number: int,
    title: str,
    fields: dict[str, str],
    source: Path,
    source_hash: str,
    locator: str,
) -> HistoricalExperimentRecord:
    context = fields.get("上下文口径", "")
    hypothesis = fields.get("假设本身", "")
    protocol = fields.get("协议", "")
    result = fields.get("结果与判定", "")
    limitations = fields.get("局限与可复用结论", fields.get("局限", ""))
    hypothesis_match = re.search(r"\b(H\d[\w_]*)\b", hypothesis)
    issues = []
    for phrase in ("缺失", "未对齐", "不一致", "bug", "符号是反的", "需重跑", "单 seed", "单切分"):
        if phrase.lower() in f"{context} {protocol} {result} {limitations}".lower():
            issues.append(phrase)
    corrections = []
    if "符号是反的" in limitations:
        corrections.append({
            "field": "residual_vs_ridge_rampdown",
            "status": "REQUIRES_SOURCE_ARTIFACT_VERIFICATION",
            "reason": "历史日志声明主字段符号相反并存在手工 corrected 值",
        })
    seeds = _ints(r"(?:种子|seeds?)\s*\[([^\]]+)\]", f"{context} {protocol}")
    tier = _tier(number, protocol, result, limitations)
    return HistoricalExperimentRecord(
        experiment_id=f"HIST-{number:03d}",
        series_id=(hypothesis_match.group(1).split("_")[0] if hypothesis_match else f"LEGACY-SERIES-{number:03d}"),
        hypothesis_id=hypothesis_match.group(1) if hypothesis_match else None,
        run_date=(re.search(r"20\d{2}-\d{2}-\d{2}", title) or re.search(r"\d{2}-\d{2}", title)).group(0) if (re.search(r"20\d{2}-\d{2}-\d{2}", title) or re.search(r"\d{2}-\d{2}", title)) else None,
        source_type="historical_markdown",
        source_path=str(source.resolve()),
        source_sha256=source_hash,
        source_locator=locator,
        scope=_scope(context, protocol),
        random_seeds=seeds,
        candidate_models=_models(f"{context} {hypothesis} {protocol} {result}"),
        locked_test_used_for_selection=False if "locked" in f"{context} {result}".lower() else None,
        verdict=_verdict(result),
        evidence_tier=tier,
        audit_status="SOURCE_ARTIFACT_REQUIRED" if tier == EvidenceTier.AUDITED_CONFIRMATORY else "LEGACY_LOG_REVIEWED",
        known_issues=issues,
        corrections=corrections,
        reproducibility_status="PARTIAL" if seeds else "UNKNOWN",
        raw_context=context,
        raw_hypothesis=hypothesis,
        raw_protocol=protocol,
        raw_result=result,
        raw_limitations=limitations,
        imported_at=datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc),
    )


def _parse_markdown(path: Path) -> tuple[list[HistoricalExperimentRecord], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    source_hash = _sha256(path)
    records: list[HistoricalExperimentRecord] = []
    issues: list[dict[str, Any]] = []
    current: tuple[int, str, int] | None = None
    fields: dict[str, str] = {}

    def flush(end_line: int) -> None:
        nonlocal current, fields
        if current is None:
            return
        number, title, start = current
        record = _record_from_section(number, title, fields, path, source_hash, f"lines:{start}-{end_line}")
        records.append(record)
        required = {"上下文口径", "假设本身", "协议", "结果与判定"}
        missing = sorted(required - set(fields))
        if missing:
            issues.append({"experiment_id": record.experiment_id, "issue": "missing_sections", "fields": missing})
        for issue in record.known_issues:
            issues.append({"experiment_id": record.experiment_id, "issue": issue, "source_locator": record.source_locator})
        current = None
        fields = {}

    for line_number, line in enumerate(lines, start=1):
        heading = _EXPERIMENT_HEADING.match(line)
        if heading:
            flush(line_number - 1)
            current = (int(heading.group(1)), heading.group(2).strip(), line_number)
            fields = {}
            for prior in reversed(lines[:line_number - 1]):
                if prior.startswith("## "):
                    break
                if prior.startswith("> 统一口径"):
                    fields["_section_context"] = prior.lstrip("> ").strip()
                    break
            continue
        if current:
            field_match = _FIELD.match(line)
            if field_match:
                key = field_match.group(1).strip()
                if key.startswith("上下文口径"):
                    key = "上下文口径"
                elif key.startswith("假设本身"):
                    key = "假设本身"
                elif key.startswith("协议"):
                    key = "协议"
                elif key.startswith("结果与判定"):
                    key = "结果与判定"
                elif key.startswith("局限"):
                    key = "局限与可复用结论"
                value = field_match.group(2).strip()
                if key == "上下文口径" and (value.startswith("如上") or value.startswith("同实验")):
                    section_context = fields.get("_section_context", "")
                    if section_context:
                        value = value + "；" + section_context
                fields[key] = value
    flush(len(lines))

    by_number = {int(record.experiment_id.split("-")[-1]): record for record in records}
    for record in records:
        reference = re.search(r"同实验\s*(\d+)", record.raw_context)
        if not reference:
            continue
        source = by_number.get(int(reference.group(1)))
        if source is None:
            continue
        inherited = source.scope.model_dump()
        inherited.update({key: value for key, value in record.scope.model_dump().items() if value not in (None, "", [])})
        record.scope = ExperimentScopeSignature.model_validate(inherited)

    planned_index = 0
    for line_number, line in enumerate(lines, start=1):
        match = _PLANNED.match(line)
        if not match or "未做" not in "\n".join(lines[max(0, line_number - 4):line_number]):
            continue
        planned_index += 1
        hypothesis_id, description = match.groups()
        records.append(HistoricalExperimentRecord(
            experiment_id=f"PLANNED-{hypothesis_id}",
            series_id=hypothesis_id.split("_")[0],
            hypothesis_id=hypothesis_id,
            source_type="historical_markdown",
            source_path=str(path.resolve()),
            source_sha256=source_hash,
            source_locator=f"line:{line_number}",
            scope=_scope(description, "未执行"),
            candidate_models=_models(description),
            verdict="NOT_EXECUTED",
            evidence_tier=EvidenceTier.PLANNED_NOT_EXECUTED,
            audit_status="NOT_EXECUTED",
            raw_hypothesis=description.strip("—- "),
            raw_protocol="未执行",
            imported_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        ))
    return records, issues


def import_experiment_history(source_path: str | Path) -> tuple[list[HistoricalExperimentRecord], list[dict[str, Any]]]:
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _parse_markdown(path)
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(f"unsupported_history_format:{suffix}")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("experiments", [payload])
    else:
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [HistoricalExperimentRecord.model_validate(item) for item in items]
    return records, []


def validate_historical_experiment(record: HistoricalExperimentRecord) -> list[str]:
    issues = list(record.known_issues)
    if record.evidence_tier == EvidenceTier.AUDITED_CONFIRMATORY:
        if not record.raw_protocol or "缺失" in record.raw_protocol:
            issues.append("confirmatory_protocol_missing")
        if not record.raw_result:
            issues.append("confirmatory_result_missing")
    if record.evidence_tier == EvidenceTier.PLANNED_NOT_EXECUTED and record.verdict not in {None, "NOT_EXECUTED"}:
        issues.append("planned_record_cannot_have_scientific_verdict")
    if record.scope.prediction_mode == "mass" and record.scope.target_variable == "steam_volumetric_flow":
        issues.append("mass_volume_scope_conflict")
    return sorted(set(issues))
