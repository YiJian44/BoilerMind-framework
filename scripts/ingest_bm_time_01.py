from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from boilermind.core.contracts import (  # noqa: E402
    EvidenceTier, ExperimentObservation, ExperimentScopeSignature,
    HistoricalExperimentRecord, ObservationType,
)
from boilermind.experiment_memory.persistence import build_empirical_capability_profile  # noqa: E402
from boilermind.experiment_memory.store import ExperimentMemoryStore  # noqa: E402

DATASET_SHA256 = "9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    artifact_root = REPO / "runtime/experiment_artifacts/BM-TIME-01"
    memory_root = REPO / "runtime/experiment_memory"
    audit_path = artifact_root / "audit.json"
    metrics_path = artifact_root / "time_block_metrics.csv"
    predictions_path = artifact_root / "time_block_predictions.csv"
    report_path = REPO / "docs/BM-TIME-01_跨时间块复验报告.md"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    if audit["status"] != "PHASE1_COMPLETED_EXPLORATORY" or len(metrics) != 24:
        raise ValueError("bm_time_01_phase1_artifacts_incomplete")
    artifacts = [audit_path, metrics_path, predictions_path, report_path]
    records, observations = [], []
    for horizon in (40, 80):
        for model in ("persistence", "ridge", "bayesianridge"):
            rows = [row for row in metrics if int(row["horizon_steps"]) == horizon and row["model"] == model]
            block_metrics = {
                row["time_block"]: {
                    key: (float(value) if key in {"mae_m3_s", "rmse_m3_s", "r2", "mbe_m3_s", "mae_improvement_vs_persistence_pct"} else value)
                    for key, value in row.items() if key not in {"experiment_id", "horizon_steps", "model"}
                }
                for row in rows
            }
            scope = ExperimentScopeSignature(
                target_variable="steam_volumetric_flow", target_definition="31特征直接预测未来蒸汽体积流量V",
                target_unit="m3/s", prediction_mode="direct_volume", dataset_id="boiler_181var_v1",
                dataset_sha256=DATASET_SHA256, feature_set_id="31_SOFT_SENSOR_FEATURES_1BASED",
                feature_count=31, window_steps=20, prediction_horizon_steps=horizon,
                sampling_interval_seconds=15, split_policy="expanding_window_time_blocks",
                metrics=["mae_m3_s", "rmse_m3_s", "r2", "mbe_m3_s"], baselines=["persistence"],
                protocol_status="PHASE1_TIME_BLOCK_SCALER_REFIT_TRAIN_ONLY",
            )
            experiment_id = f"BM-TIME-01-PHASE1-{model.upper()}-H{horizon}"
            issues = ["PHASE1_LINEAR_MODELS_ONLY", "TORCH_AND_RF_TIME_BLOCK_REPLICATION_PENDING", "HUMAN_REVIEW_PENDING"]
            record = HistoricalExperimentRecord(
                experiment_id=experiment_id, series_id="BM-TIME-01",
                parent_experiment_ids=[f"BM-SEED-02-{model.upper()}-H{horizon}-S7-19-42"],
                hypothesis_id=f"H-BM-TIME-01-{model.upper()}-H{horizon}", run_date="2026-08-22",
                source_type="deterministic_expanding_window_retraining", source_path=str(audit_path),
                source_sha256=sha256(audit_path), source_locator=f"phase1:model={model},horizon={horizon}",
                scope=scope, random_seeds=[42], protocol_path=str(REPO / "scripts/run_bm_time_01.py"),
                candidate_models=[model], selection_scope="ridge_alpha_validation_only_per_time_block; locked_test_block_evaluation",
                locked_test_used_for_selection=False,
                confirmation_criteria=["torch_and_rf_extension", "human_review"],
                metrics={"time_blocks": block_metrics}, verdict="TIME_BLOCK_RANKING_DEPENDENCE_OBSERVED_PHASE1",
                verdict_scope=["31v_direct", f"h{horizon}", model, "expanding_time_blocks", "seed42"],
                evidence_tier=EvidenceTier.AUDITED_EXPLORATORY,
                audit_status="PASSED_PHASE1_TIME_BLOCK_RECOMPUTATION", known_issues=issues,
                reproducibility_status="HASHED_METRICS_PREDICTIONS_AUDIT_AND_REPORT_ARCHIVED",
                artifact_paths=[str(path) for path in artifacts], artifact_hashes={str(path): sha256(path) for path in artifacts},
                raw_context="BM-TIME-01第一阶段跨时间块重训；每块重新使用训练段拟合特征缩放器。",
                raw_hypothesis=f"{model}在h{horizon}任务上的相对性能是否跨时间块保持稳定。",
                raw_protocol="early/middle/late/latest_holdout; expanding train; block-specific train-only MinMax; seed42.",
                raw_result=json.dumps(block_metrics, ensure_ascii=False, sort_keys=True), raw_limitations="; ".join(issues),
                importer_version="1.0.0",
            )
            records.append(record)
            observations.append(ExperimentObservation(
                observation_id=f"OBS-{experiment_id}-TIME-BOUNDARY", source_experiment_ids=[experiment_id],
                observation_type=ObservationType.BOUNDARY_CONDITION,
                claim=f"{model}在direct-volume h{horizon}的表现具有明显时间块边界；第一阶段结果不能外推为跨全时段稳定能力。",
                scope_signature=scope, comparison_signature=scope.model_dump_json(exclude_none=True),
                supporting_metrics=block_metrics, confidence_level=0.84,
                reuse_policy="EXACT_SCOPE_TIME_BOUNDARY_ONLY; PHASE2_AND_HUMAN_REVIEW_REQUIRED",
                invalid_for_scientific_synthesis=False, derived_by="bm_time_01_phase1_ingester", derivation_version="1.0.0",
            ))
    store = ExperimentMemoryStore(memory_root)
    old_records, old_observations = store.load_records(), store.load_observations()
    record_ids, observation_ids = {r.experiment_id for r in old_records}, {o.observation_id for o in old_observations}
    merged_records = old_records + [r for r in records if r.experiment_id not in record_ids]
    merged_observations = old_observations + [o for o in observations if o.observation_id not in observation_ids]
    issues = [json.loads(line) for line in store.issues_path.read_text(encoding="utf-8").splitlines() if line.strip()] if store.issues_path.is_file() else []
    store.replace_all(merged_records, merged_observations, issues)
    (store.root / "empirical_capability_profile.json").write_text(build_empirical_capability_profile(merged_records).model_dump_json(indent=2), encoding="utf-8")
    result = {"schema_version": "boilermind.bm_time_01_phase1_ingestion.v1", "records_added_or_verified": 6,
              "observations_added_or_verified": 6, "memory_total_records": len(merged_records),
              "memory_total_observations": len(merged_observations), "evidence_tier": "AUDITED_EXPLORATORY",
              "phase2_status": "TORCH_AND_RF_PENDING"}
    (store.root / "bm_time_01_phase1_ingestion_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
