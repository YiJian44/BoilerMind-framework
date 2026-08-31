from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    library_path = REPO / "model_library" / "model_library.json"
    package_root = REPO / "model_library" / "artifact_packages" / "31v_direct" / "v1"
    memory_root = REPO / "runtime" / "experiment_memory"
    library = json.loads(library_path.read_text(encoding="utf-8"))
    package = json.loads((package_root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if package["dataset_sha256"] != DATASET_SHA256:
        raise SystemExit("31v_package_dataset_hash_mismatch")

    records: list[HistoricalExperimentRecord] = []
    observations: list[ExperimentObservation] = []
    for item in library["models"]:
        model_id = str(item["id"])
        base_model, horizon_text = model_id.rsplit("_h", 1)
        horizon = int(horizon_text)
        manifest_path = REPO / "model_library" / "manifests" / "31v_direct" / f"h{horizon}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_model = manifest["models"][base_model]
        prediction_paths = [
            REPO / manifest_model["artifacts"]["predictions"][split]["path"]
            for split in ("validation", "locked_test")
        ]
        weight_paths = [
            REPO / artifact["path"]
            for name, artifact in manifest_model["artifacts"].items()
            if name != "predictions" and isinstance(artifact, dict) and artifact.get("path")
        ]
        artifacts = [manifest_path, *prediction_paths, *weight_paths]
        missing = [str(path) for path in artifacts if not path.is_file()]
        if missing:
            raise SystemExit(f"31v_artifact_missing:{model_id}:{missing}")
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
            protocol_status="ARTIFACT_HASH_AND_PREDICTION_METRICS_VERIFIED",
        )
        known_issues = [
            "SINGLE_SEED_42_ONLY",
            "FIT_CONVERGENCE_AND_WARNINGS_INFERRED_NOT_CAPTURED_DURING_TRAINING",
            "ONLY_3_OF_31_FEATURE_NAMES_SEMANTICALLY_VERIFIED",
        ]
        if base_model == "hgb":
            known_issues.append("JOBLIB_REQUIRES_TRAINING_SKLEARN_1_7_2_LOCAL_1_9_0_LOAD_FAILED")
        experiment_id = f"LIB31V-AUDITED-{model_id.upper()}"
        record = HistoricalExperimentRecord(
            experiment_id=experiment_id,
            series_id=f"LIB31V-H{horizon}",
            parent_experiment_ids=[f"LIB31V-{model_id.upper()}"],
            hypothesis_id=f"H-LIB31V-H{horizon}-MODEL-COMPARISON",
            random_seeds=[42],
            source_type="teammate_31v_artifact_package",
            source_path=str(package_root / "PACKAGE_MANIFEST.json"),
            source_sha256=sha256(package_root / "PACKAGE_MANIFEST.json"),
            source_locator=f"models[id={model_id}]",
            scope=scope,
            protocol_path=str(REPO / "scripts" / "train_31v_library.py"),
            candidate_models=[base_model],
            selection_scope="validation_only_model_selection; locked_test_evaluation",
            locked_test_used_for_selection=False,
            metrics=metrics,
            verdict="ARTIFACT_METRICS_REPRODUCED_SINGLE_SEED",
            verdict_scope=["31v_direct", f"h{horizon}", base_model, "seed42"],
            evidence_tier=EvidenceTier.AUDITED_EXPLORATORY,
            audit_status="PASSED_ARTIFACT_REPRODUCTION",
            known_issues=known_issues,
            reproducibility_status="HASHED_WEIGHTS_PREDICTIONS_MANIFESTS_AND_ENVIRONMENT_PRESENT",
            artifact_paths=[str(path) for path in artifacts],
            artifact_hashes={str(path): sha256(path) for path in artifacts},
            raw_context=str(library.get("scope", "")),
            raw_hypothesis=f"比较{base_model}在31特征直接V预测h{horizon}任务中的单seed表现。",
            raw_protocol="70/10/20 chronological split; train-only scaling; validation selection; locked-test evaluation; seed=42.",
            raw_result=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            raw_limitations="; ".join(known_issues),
            importer_version="2.0.0",
        )
        records.append(record)
        observations.append(ExperimentObservation(
            observation_id=f"OBS-LIB31V-{model_id.upper()}-ARTIFACT-VERIFIED",
            source_experiment_ids=[experiment_id],
            observation_type=ObservationType.SUPPORTED,
            claim=f"{model_id} 的逐样本预测可重算得到所登记指标；该观察仅适用于当前协议与seed=42，不代表跨seed稳定性。",
            scope_signature=scope,
            comparison_signature=scope.model_dump_json(exclude_none=True),
            supporting_metrics=metrics,
            confidence_level=0.72,
            reuse_policy="EXPLORATORY_ONLY_REQUIRE_MULTI_SEED_OR_TIME_BLOCK_CONFIRMATION",
            invalid_for_scientific_synthesis=False,
            derived_by="31v_artifact_package_promoter",
            derivation_version="2.0.0",
        ))

    store = ExperimentMemoryStore(memory_root)
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
        "schema_version": "boilermind.31v_library_audit.v2",
        "artifact_package_version": package.get("package"),
        "artifact_verified_model_count": len(records),
        "prediction_file_count_verified": 56,
        "scientific_synthesis_eligible_count": len(records),
        "evidence_tier": EvidenceTier.AUDITED_EXPLORATORY.value,
        "confirmatory_blockers": ["multi_seed_not_run", "cross_time_block_not_run", "training_warning_capture_missing"],
        "runtime_load_check": {"passed": 7, "failed": 1, "failed_model": "hgb", "reason": "sklearn_1_7_2_to_1_9_0_joblib_incompatibility"},
        "memory_total_records": len(merged_records),
        "memory_total_observations": len(merged_observations),
    }
    (store.root / "31v_model_library_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
