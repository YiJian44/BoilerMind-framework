from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from boilermind.core.contracts import (
    EvidenceTier,
    ExperimentObservation,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
    ObservationType,
)
from boilermind.experiment_memory.persistence import build_empirical_capability_profile
from boilermind.experiment_memory.store import ExperimentMemoryStore


DATASET_SHA256 = "9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="将31/V direct模型库元数据安全写入实验记忆。")
    parser.add_argument("--library", default="model_library/model_library.json")
    parser.add_argument("--dataset", default="resources/datasets/boiler_181var_v1/boiler_181var_clean.csv")
    parser.add_argument("--memory-root", default="runtime/experiment_memory")
    args = parser.parse_args()
    library_path = Path(args.library).resolve()
    dataset_path = Path(args.dataset).resolve()
    if _sha256(dataset_path) != DATASET_SHA256:
        raise SystemExit("31v_dataset_sha256_mismatch")
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    if payload.get("count") != len(payload.get("models", [])):
        raise SystemExit("31v_library_count_mismatch")
    library_hash = _sha256(library_path)
    weights_root = library_path.parent / "weights"
    cache_root = Path("runtime/31v_data").resolve()
    records: list[HistoricalExperimentRecord] = []
    observations: list[ExperimentObservation] = []
    missing_weight_ids: list[str] = []
    for item in payload["models"]:
        model_id = str(item["id"])
        base_model, horizon_text = model_id.rsplit("_h", 1)
        horizon = int(horizon_text)
        weights = item.get("weights") or {}
        declared_weight = bool(weights.get("exists"))
        weight_file = weights_root / str(weights.get("dir") or "") / str(weights.get("weight_file") or "")
        local_weight = declared_weight and weight_file.is_file()
        if declared_weight and not local_weight:
            missing_weight_ids.append(model_id)
        issues = [
            "METADATA_ONLY_IMPORT",
            "PER_SAMPLE_PREDICTIONS_NOT_AVAILABLE",
            "SOURCE_TRAINING_MANIFEST_NOT_AVAILABLE",
        ]
        if declared_weight and not local_weight:
            issues.append("DECLARED_SERVER_WEIGHT_NOT_PRESENT_LOCALLY")
        if not cache_root.is_dir():
            issues.append("WINDOWED_DATA_CACHE_NOT_PRESENT_LOCALLY")
        metrics = item["metrics"]
        scope = ExperimentScopeSignature(
            target_variable="steam_volumetric_flow",
            target_definition="31特征直接预测未来蒸汽体积流量V",
            target_unit="m3/s",
            prediction_mode="direct-V",
            dataset_id="boiler_181var_v1",
            dataset_sha256=DATASET_SHA256,
            feature_set_id="31_SOFT_SENSOR_FEATURES_1BASED",
            feature_count=31,
            window_steps=20,
            prediction_horizon_steps=horizon,
            sampling_interval_seconds=15,
            split_policy="chronological_train_validation_locked_test",
            split_ratios=[0.70, 0.10, 0.20],
            metrics=list(metrics["locked_test"]),
            baselines=["persistence"],
            protocol_status="METADATA_ONLY_PENDING_ARTIFACT_AUDIT",
        )
        record = HistoricalExperimentRecord(
            experiment_id=f"LIB31V-{model_id.upper()}",
            series_id=f"LIB31V-H{horizon}",
            hypothesis_id=f"H-LIB31V-H{horizon}-MODEL-COMPARISON",
            source_type="teammate_model_library_metadata",
            source_path=str(library_path),
            source_sha256=library_hash,
            source_locator=f"models[id={model_id}]",
            scope=scope,
            protocol_path=str(Path("scripts/train_31v_library.py").resolve()),
            candidate_models=[base_model],
            selection_scope="validation_only_model_selection; locked_test_evaluation",
            locked_test_used_for_selection=False,
            metrics=metrics,
            verdict="BENCHMARK_METADATA_PENDING_ARTIFACT_AUDIT",
            verdict_scope=["31v_direct", f"h{horizon}", base_model],
            evidence_tier=EvidenceTier.LEGACY_INFORMATIVE,
            audit_status="SOURCE_ARTIFACT_REQUIRED",
            known_issues=issues,
            reproducibility_status="TRAINING_CODE_AND_DATA_PRESENT_WEIGHTS_AND_PREDICTIONS_MISSING",
            artifact_paths=[str(library_path), str(dataset_path), str(Path("scripts/train_31v_library.py").resolve())],
            artifact_hashes={str(library_path): library_hash, str(dataset_path): DATASET_SHA256},
            raw_context=str(payload.get("scope", "")),
            raw_hypothesis=f"比较{base_model}在31特征直接V预测h{horizon}任务中的表现。",
            raw_protocol="70/10/20 chronological split; train-only scaling; validation selection; locked-test evaluation.",
            raw_result=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            raw_limitations="; ".join(issues),
        )
        records.append(record)
        claim = (
            f"{model_id}登记的locked-test指标为"
            + json.dumps(metrics["locked_test"], ensure_ascii=False, sort_keys=True)
            + "；因本地缺少权重、逐样本预测和完整训练manifest，当前仅作待审计元数据。"
        )
        observations.append(ExperimentObservation(
            observation_id=f"OBS-LIB31V-{model_id.upper()}-PENDING-AUDIT",
            source_experiment_ids=[record.experiment_id],
            observation_type=ObservationType.DATA_QUALITY_WARNING,
            claim=claim,
            scope_signature=scope,
            comparison_signature=scope.model_dump_json(exclude_none=True),
            supporting_metrics=metrics,
            confidence_level=0.55,
            reuse_policy="SOURCE_ARTIFACT_VERIFICATION_REQUIRED",
            invalid_for_scientific_synthesis=True,
            derived_by="31v_model_library_metadata_ingester",
            derivation_version="1.0.0",
        ))
    store = ExperimentMemoryStore(args.memory_root)
    existing_records = store.load_records()
    existing_observations = store.load_observations()
    record_ids = {item.experiment_id for item in existing_records}
    observation_ids = {item.observation_id for item in existing_observations}
    merged_records = existing_records + [item for item in records if item.experiment_id not in record_ids]
    merged_observations = existing_observations + [item for item in observations if item.observation_id not in observation_ids]
    issues = [json.loads(line) for line in store.issues_path.read_text(encoding="utf-8").splitlines() if line.strip()] if store.issues_path.is_file() else []
    store.replace_all(merged_records, merged_observations, issues)
    profile = build_empirical_capability_profile(merged_records)
    (store.root / "empirical_capability_profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    report = {
        "schema_version": "boilermind.31v_library_audit.v1",
        "library_version": payload.get("library_version"),
        "library_sha256": library_hash,
        "dataset_sha256": DATASET_SHA256,
        "record_count": len(records),
        "observation_count": len(observations),
        "local_weights_root_present": weights_root.is_dir(),
        "local_window_cache_present": cache_root.is_dir(),
        "declared_weights_missing_locally": missing_weight_ids,
        "scientific_synthesis_eligible_count": 0,
        "runtime_executable_capability_added": False,
        "reason": "weights_predictions_and_training_manifests_not_present_locally",
        "memory_total_records": len(merged_records),
        "memory_total_observations": len(merged_observations),
    }
    (store.root / "31v_model_library_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
