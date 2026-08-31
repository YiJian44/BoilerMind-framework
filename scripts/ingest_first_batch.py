from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boilermind.core.contracts import (
    EvidenceTier,
    ExperimentObservation,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
    ObservationType,
)
from boilermind.experiment_memory.persistence import build_empirical_capability_profile
from boilermind.experiment_memory.store import ExperimentMemoryStore


VALID_EXPLORATORY = {
    "ridge", "bayesianridge", "svr", "mlp", "pls", "hgb", "rf", "knn",
}
PROTOCOL_INVALID = {"dlinear", "lstm", "gru", "transformer"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _obs(record: HistoricalExperimentRecord, suffix: str, kind: ObservationType,
         claim: str, confidence: float, *, invalid: bool = False,
         metrics: dict[str, Any] | None = None) -> ExperimentObservation:
    return ExperimentObservation(
        observation_id=f"OBS-{record.experiment_id}-{suffix}",
        source_experiment_ids=[record.experiment_id],
        observation_type=kind,
        claim=claim,
        scope_signature=record.scope,
        comparison_signature=record.scope.model_dump_json(exclude_none=True),
        supporting_metrics=metrics or {},
        confidence_level=confidence,
        reuse_policy="ENGINEERING_ONLY" if invalid else "SCOPE_MATCH_REQUIRED_SINGLE_SEED",
        invalid_for_scientific_synthesis=invalid,
        derived_by="first_batch_structured_ingester",
        derivation_version="1.0.0",
    )


def _scope(dataset_hash: str | None = None) -> ExperimentScopeSignature:
    return ExperimentScopeSignature(
        target_variable="main_steam_mass_flow",
        target_definition="10分钟后主蒸汽质量流量",
        target_unit="t/h",
        prediction_mode="M",
        dataset_id="boiler_real_data_first_batch",
        dataset_sha256=dataset_hash,
        feature_set_id="all_30_features",
        feature_count=30,
        window_steps=20,
        prediction_horizon_steps=40,
        sampling_interval_seconds=15,
        split_policy="chronological_train_validation_locked_test",
        split_ratios=[0.70, 0.15, 0.15],
        metrics=["mae_t_h", "rmse_t_h", "r2", "mbe_t_h"],
        baselines=["persistence"],
        protocol_status="SINGLE_SEED_EXPLORATORY",
    )


def _completed_record(row: dict[str, Any], result_path: Path) -> tuple[HistoricalExperimentRecord, list[ExperimentObservation]]:
    model = row["model"]
    invalid = model in PROTOCOL_INVALID
    tier = EvidenceTier.ENGINEERING_FAILURE if invalid else EvidenceTier.AUDITED_EXPLORATORY
    audit_status = "FAILED_PROTOCOL_AUDIT" if invalid else "PASSED_EXPLORATORY_ONLY"
    locked = row["locked_test_metrics"]
    baseline = row["persistence_locked_test_metrics"]
    improvement = row["locked_test_mae_improvement_ratio"]
    warnings = list(row.get("warnings") or [])
    issues = list(warnings)
    if invalid:
        issues.extend([
            "TARGET_SCALING_OR_TRAINING_PROTOCOL_NOT_VALIDATED",
            "RESULT_INVALID_FOR_SCIENTIFIC_MODEL_COMPARISON",
        ])
    if model == "mlp":
        issues.append("OPTIMIZER_DID_NOT_CONVERGE_WITHIN_300_ITERATIONS")
    if model == "hgb":
        issues.append("ENVIRONMENT_WARNING_WAS_MISCLASSIFIED_AS_NON_CONVERGENCE_BY_RUNNER")
    artifact_paths = [result_path]
    for name in ("contract.json", "manifest.json", "metrics/model_metrics.json", "logs/execution.json"):
        candidate = result_path.parent / name
        if candidate.is_file():
            artifact_paths.append(candidate)
    record = HistoricalExperimentRecord(
        experiment_id=row["experiment_id"],
        series_id="BM-SEED-01",
        hypothesis_id="H-SEED-01",
        run_date="2026-08-21",
        source_type="boilermind_unified_runner",
        source_path=str(result_path),
        source_sha256=_sha256(result_path),
        source_locator=f"result:{result_path}",
        scope=_scope(next(iter(json.loads(result_path.read_text(encoding="utf-8"))["model_records"].values()))["dataset_sha256"]),
        random_seeds=[42],
        protocol_path=str(result_path.parent / "contract.json"),
        candidate_models=[model],
        selection_scope="validation_only; independent per-model run; cross-model ranking is post-hoc exploratory",
        locked_test_used_for_selection=False,
        confirmation_criteria=["independent_seed_replication_required_before_confirmatory_claim"],
        metrics={
            "validation": row["validation_metrics"],
            "locked_test": locked,
            "persistence_locked_test": baseline,
            "locked_test_mae_improvement_ratio": improvement,
            "runtime_seconds": row["runtime_seconds"],
        },
        verdict="PROTOCOL_INVALID" if invalid else "EXPLORATORY_OBSERVATION",
        verdict_scope=["single_seed_42", "M@40", "window_20", model],
        evidence_tier=tier,
        audit_status=audit_status,
        known_issues=sorted(set(issues)),
        reproducibility_status="ARTIFACTS_CAPTURED_SINGLE_SEED",
        artifact_paths=[str(path) for path in artifact_paths],
        artifact_hashes={str(path): _sha256(path) for path in artifact_paths},
        raw_context="第一批真实数据、统一时序切分、单随机种子模型基线实验。",
        raw_hypothesis="候选模型在锁定测试集上相对 persistence 的误差表现存在可验证差异。",
        raw_protocol="M@40, window=20, 15s, chronological 70/15/15, validation-only selection, locked test.",
        raw_result=json.dumps({"model": model, "locked_test": locked, "baseline": baseline, "improvement": improvement}, ensure_ascii=False),
        raw_limitations="; ".join(sorted(set(issues + ["SINGLE_SEED_NOT_CONFIRMATORY", "PREDICTIONS_NOT_PERSISTED"]))),
    )
    if invalid:
        claim = f"{model} 本次调用虽返回结果，但目标尺度/训练协议未通过审计，数值不得用于模型优劣结论。"
        return record, [_obs(record, "PROTOCOL", ObservationType.ENGINEERING_FAILURE, claim, 0.95, invalid=True, metrics=record.metrics)]
    if improvement is not None and improvement > 0:
        kind, direction = ObservationType.SUPPORTED, "低于"
    else:
        kind, direction = ObservationType.FALSIFIED, "未低于"
    claim = (
        f"在单种子42、M@40、window=20的锁定测试集上，{model} 的MAE为"
        f"{locked['mae_t_h']:.6f} t/h，{direction} persistence 的{baseline['mae_t_h']:.6f} t/h；"
        "该观察仅限探索性使用，需独立种子复验。"
    )
    observations = [_obs(record, "BASELINE", kind, claim, 0.65, metrics=record.metrics)]
    if model == "mlp":
        observations.append(_obs(record, "CONVERGENCE", ObservationType.DATA_QUALITY_WARNING,
                                 "MLP 在300次迭代内未收敛，本次数值可记录但不应作为稳定能力上限。", 0.95,
                                 invalid=True))
    if model == "hgb":
        observations.append(_obs(record, "WARNING", ObservationType.DATA_QUALITY_WARNING,
                                 "HGB 的物理核心数环境警告被运行器误判为未收敛；需修复警告分类。", 0.95,
                                 invalid=True))
    return record, observations


def _incomplete_records(summary_path: Path) -> list[tuple[HistoricalExperimentRecord, list[ExperimentObservation]]]:
    source_hash = _sha256(summary_path)
    rows = []
    elastic = HistoricalExperimentRecord(
        experiment_id="BM-SEED-01-ELASTICNET-S42",
        series_id="BM-SEED-01", hypothesis_id="H-SEED-01", run_date="2026-08-21",
        source_type="boilermind_operator_observation", source_path=str(summary_path),
        source_sha256=source_hash, source_locator="not_completed:model=elasticnet",
        scope=_scope(), random_seeds=[42], candidate_models=["elasticnet"],
        verdict="ENGINEERING_TIMEOUT", evidence_tier=EvidenceTier.ENGINEERING_FAILURE,
        audit_status="FAILED_TO_COMPLETE", known_issues=["FIT_EXCEEDED_FIRST_BATCH_OPERATIONAL_BUDGET", "NO_PROCESS_LEVEL_HARD_TIMEOUT"],
        reproducibility_status="TIMEOUT_OBSERVED_NO_RESULT_ARTIFACT",
        raw_result="ElasticNet fit was manually interrupted after exceeding the first-batch operational budget.",
        raw_limitations="No result metrics; this is an engineering failure, not scientific falsification.",
    )
    rows.append((elastic, [_obs(elastic, "TIMEOUT", ObservationType.ENGINEERING_FAILURE,
        "ElasticNet 本轮拟合超过运行预算并被中止；这是工程超时，不是模型假设被证伪。", 0.95, invalid=True)]))
    gpr = HistoricalExperimentRecord(
        experiment_id="BM-SEED-01-GPR-S42",
        series_id="BM-SEED-01", hypothesis_id="H-SEED-01-GPR-FEASIBILITY", run_date="2026-08-21",
        source_type="boilermind_preregistered_but_blocked", source_path=str(summary_path),
        source_sha256=source_hash, source_locator="not_completed:model=gpr", scope=_scope(),
        random_seeds=[42], candidate_models=["gpr"], verdict="NOT_EXECUTED_RESOURCE_SAFETY_BLOCK",
        evidence_tier=EvidenceTier.PLANNED_NOT_EXECUTED, audit_status="NOT_EXECUTED",
        known_issues=["EXACT_GPR_O_N3_RISK", "NO_HARD_MEMORY_OR_TIME_ISOLATION"],
        reproducibility_status="NOT_EXECUTED",
        raw_result="Exact GPR was not launched on approximately 17600 training windows.",
        raw_limitations="Planned-only record; generates no scientific observation.",
    )
    rows.append((gpr, []))
    return rows


def _write_capability_markdown(profile, destination: Path, first_batch_records: list[HistoricalExperimentRecord]) -> None:
    batch_ids = {item.experiment_id for item in first_batch_records}
    lines = [
        "# BoilerMind 实证实验能力档案", "",
        f"更新时间：{datetime.now(timezone.utc).isoformat()}", "",
        "> 本文档描述历史运行证据，不等同于注册表宣称的可执行能力。单种子探索结果不能升级为确认性结论。", "",
        "## 第一批入库结论", "",
        "- 有效探索性运行：8（Ridge、Bayesian Ridge、SVR、MLP、PLS、HGB、RF、KNN）。",
        "- 协议/数值失效：4（DLinear、LSTM、GRU、Transformer），仅作工程诊断。",
        "- 工程超时：1（ElasticNet），不是科学证伪。",
        "- 计划但未执行：1（精确 GPR），因资源安全门阻断。", "",
        "## 模型能力汇总", "",
        "| 模型 | 历史记录数 | 成功记录 | 工程失败 | 收敛失败 | 已记录耗时(s) | 置信度 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for model in profile.models:
        runtimes = ", ".join(f"{value:.3f}" for value in model.runtime_seconds) or "—"
        lines.append(f"| {model.model_id} | {model.run_count} | {model.success_count} | {model.failure_count} | {model.convergence_failure_count} | {runtimes} | {model.confidence:.2f} |")
    lines.extend(["", "## 第一批记录索引", ""])
    for record in first_batch_records:
        lines.append(f"- `{record.experiment_id}`：{record.evidence_tier.value} / {record.audit_status} / {record.verdict}")
    lines.extend(["", "## 当前能力边界", "",
        "- 第一批只验证了 M@40、window=20、15秒采样、固定数据哈希下的单种子路径。",
        "- 尚未验证跨种子稳定性、跨时段漂移、工况分层、置信区间和直接体积流量任务。",
        "- 当前运行器未保存逐样本预测与索引，不能直接做工况复用分析。",
        "- Torch 路径需完成目标缩放/反缩放和训练有效性修复后再重跑。",
        "- `fit_success` 只表示函数返回，不能单独作为科学有效或能力成功依据。", "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="将第一批真实实验分层写入 BoilerMind 实验记忆。")
    parser.add_argument("--summary", default="outputs/first_batch/BM-SEED-01-summary.json")
    parser.add_argument("--memory-root", default="runtime/experiment_memory")
    args = parser.parse_args()
    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    store = ExperimentMemoryStore(args.memory_root)
    additions: list[tuple[HistoricalExperimentRecord, list[ExperimentObservation]]] = []
    for row in summary["ranking_by_locked_test_mae"]:
        additions.append(_completed_record(row, Path(row["result_path"])))
    additions.extend(_incomplete_records(summary_path))
    existing_records = store.load_records()
    existing_observations = store.load_observations()
    new_records = [record for record, _ in additions]
    new_observations = [obs for _, observations in additions for obs in observations]
    existing_ids = {record.experiment_id for record in existing_records}
    records = existing_records + [record for record in new_records if record.experiment_id not in existing_ids]
    existing_obs_ids = {obs.observation_id for obs in existing_observations}
    observations = existing_observations + [obs for obs in new_observations if obs.observation_id not in existing_obs_ids]
    issues = [json.loads(line) for line in store.issues_path.read_text(encoding="utf-8").splitlines() if line.strip()] if store.issues_path.is_file() else []
    store.replace_all(records, observations, issues)
    profile = build_empirical_capability_profile(records)
    (store.root / "empirical_capability_profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    _write_capability_markdown(profile, store.root / "EMPIRICAL_CAPABILITY.md", new_records)
    report = {
        "schema_version": "boilermind.first_batch_ingestion.v1",
        "source_summary": str(summary_path),
        "records_added_or_verified": len(new_records),
        "observations_added_or_verified": len(new_observations),
        "evidence_tier_counts": {tier.value: 0 for tier in EvidenceTier},
        "observation_type_counts": {kind.value: 0 for kind in ObservationType},
        "total_memory_records": len(records),
        "total_memory_observations": len(observations),
    }
    for record in new_records:
        report["evidence_tier_counts"][record.evidence_tier.value] = report["evidence_tier_counts"].get(record.evidence_tier.value, 0) + 1
    for obs in new_observations:
        report["observation_type_counts"][obs.observation_type.value] = report["observation_type_counts"].get(obs.observation_type.value, 0) + 1
    (store.root / "first_batch_ingestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
