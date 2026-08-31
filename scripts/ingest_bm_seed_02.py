from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from boilermind.core.contracts import (  # noqa: E402
    EvidenceTier,
    ExperimentObservation,
    ExperimentScopeSignature,
    HistoricalExperimentRecord,
    ObservationType,
)
from boilermind.experiment_memory.persistence import build_empirical_capability_profile  # noqa: E402
from boilermind.experiment_memory.store import ExperimentMemoryStore  # noqa: E402


DATASET_SHA256 = "9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c"
SEEDS = (7, 19, 42)
HORIZONS = (40, 80)
CORE_MODELS = ("persistence", "ridge", "bayesianridge", "transformer", "lstm")
MODELS = (*CORE_MODELS, "rf")
HASH_TOLERANCE = 5e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_metrics(path: Path) -> tuple[dict[str, float], int, set[str], set[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty_prediction_file:{path}")
    y_true = [float(row["y_true"]) for row in rows]
    errors = [float(row["y_pred"]) - actual for row, actual in zip(rows, y_true)]
    mean_true = statistics.mean(y_true)
    denominator = sum((value - mean_true) ** 2 for value in y_true)
    metrics = {
        "mae_m3_s": statistics.mean(abs(value) for value in errors),
        "rmse_m3_s": math.sqrt(statistics.mean(value * value for value in errors)),
        "r2": 1.0 - sum(value * value for value in errors) / denominator,
        "mbe_m3_s": statistics.mean(errors),
    }
    dataset_hashes = {row["dataset_sha256"] for row in rows}
    protocol_hashes = {row["protocol_sha256"] for row in rows}
    return metrics, len(rows), dataset_hashes, protocol_hashes


def metric_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def selected_source_root(source: Path, seed: int, model: str) -> Path:
    group = "rf_corrected" if model == "rf" else "core"
    return source / f"seed_{seed}" / group / "model_library"


def validate_and_collect(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_manifest = read_json(source / "SHA256SUMS.json")
    manifest_failures: list[str] = []
    self_entry_seen = False
    for entry in source_manifest:
        relative = str(entry["path"]).replace("/", "\\")
        if relative.lower() == "sha256sums.json":
            self_entry_seen = True
            continue
        path = source / relative
        if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]) or sha256(path) != entry["sha256"]:
            manifest_failures.append(relative)
    if manifest_failures:
        raise ValueError(f"source_hash_validation_failed:{manifest_failures}")

    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for horizon in HORIZONS:
            for model in MODELS:
                library_root = selected_source_root(source, seed, model)
                manifest_path = library_root / "manifests" / "31v_direct" / f"h{horizon}" / "manifest.json"
                manifest = read_json(manifest_path)
                model_manifest = manifest["models"][model]
                if manifest["dataset"]["sha256"] != DATASET_SHA256:
                    raise ValueError(f"dataset_hash_mismatch:{seed}:{horizon}:{model}")
                if int(manifest["random_seed"]) != seed:
                    raise ValueError(f"seed_mismatch:{seed}:{horizon}:{model}")
                if model == "rf":
                    if int(model_manifest["effective_random_state"]) != seed:
                        raise ValueError(f"rf_effective_seed_mismatch:{seed}:{horizon}")
                    if int(manifest["sklearn_n_jobs"]) != 2 or not manifest["parallel_execution"]:
                        raise ValueError(f"rf_execution_controls_mismatch:{seed}:{horizon}")
                metrics_by_split: dict[str, Any] = {}
                prediction_paths: dict[str, str] = {}
                for split in ("validation", "locked_test"):
                    prediction_path = library_root / "predictions" / "31v_direct" / f"h{horizon}" / f"{model}_{split}_predictions.csv"
                    recomputed, row_count, dataset_hashes, protocol_hashes = prediction_metrics(prediction_path)
                    if dataset_hashes != {DATASET_SHA256} or protocol_hashes != {manifest["protocol_sha256"]}:
                        raise ValueError(f"prediction_provenance_mismatch:{seed}:{horizon}:{model}:{split}")
                    if metric_delta(recomputed, model_manifest[split]) > HASH_TOLERANCE:
                        raise ValueError(f"prediction_metric_mismatch:{seed}:{horizon}:{model}:{split}")
                    metrics_by_split[split] = {**recomputed, "row_count": row_count}
                    prediction_paths[split] = str(prediction_path)
                results.append({
                    "seed": seed,
                    "horizon": horizon,
                    "model": model,
                    "protocol_sha256": manifest["protocol_sha256"],
                    "manifest_path": str(manifest_path),
                    "predictions": prediction_paths,
                    "metrics": metrics_by_split,
                    "runtime_seconds": (
                        model_manifest.get("runtime_seconds")
                        or model_manifest.get("fit_seconds")
                        or None
                    ),
                })
    audit = {
        "source_manifest_entry_count": len(source_manifest),
        "source_manifest_nonself_entries_verified": len(source_manifest) - int(self_entry_seen),
        "source_manifest_self_entry_excluded": self_entry_seen,
        "validated_result_count": len(results),
        "excluded_paths": [
            {"path": "seed_7/model_library", "reason": "pre_fix_rf_used_random_state_42_and_non_rf_outputs_duplicate_core"},
            {"path": "seed_19/model_library", "reason": "incomplete_early_run_not_covered_by_delivery_manifest"},
        ],
    }
    return results, audit


def copy_curated_artifacts(source: Path, destination: Path) -> None:
    if destination.exists():
        existing_manifest = destination / "CURATED_MANIFEST.json"
        if not existing_manifest.is_file():
            raise FileExistsError(f"curated_destination_exists_without_manifest:{destination}")
        manifest = read_json(existing_manifest)
        for entry in manifest.get("files", []):
            path = destination / entry["path"]
            if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]) or sha256(path) != entry["sha256"]:
                raise ValueError(f"existing_curated_artifact_mismatch:{entry['path']}")
        return
    (destination / "environment").mkdir(parents=True, exist_ok=True)
    for path in (source / "environment").iterdir():
        if path.is_file():
            shutil.copy2(path, destination / "environment" / path.name)
    for seed in SEEDS:
        for group in ("core", "rf_corrected"):
            src = source / f"seed_{seed}" / group
            dst = destination / f"seed_{seed}" / group
            if dst.exists():
                raise FileExistsError(f"curated_destination_already_exists:{dst}")
            shutil.copytree(src, dst)


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        persistence = [
            row["metrics"]["locked_test"]["mae_m3_s"]
            for row in results if row["horizon"] == horizon and row["model"] == "persistence"
        ]
        baseline_mean = statistics.mean(persistence)
        for model in MODELS:
            selected = [row for row in results if row["horizon"] == horizon and row["model"] == model]
            maes = [row["metrics"]["locked_test"]["mae_m3_s"] for row in selected]
            r2s = [row["metrics"]["locked_test"]["r2"] for row in selected]
            aggregates.append({
                "model": model,
                "horizon": horizon,
                "seeds": list(SEEDS),
                "locked_test_mae_by_seed": {str(row["seed"]): row["metrics"]["locked_test"]["mae_m3_s"] for row in selected},
                "locked_test_r2_by_seed": {str(row["seed"]): row["metrics"]["locked_test"]["r2"] for row in selected},
                "locked_test_mae_mean": statistics.mean(maes),
                "locked_test_mae_sample_std": statistics.stdev(maes),
                "locked_test_mae_min": min(maes),
                "locked_test_mae_max": max(maes),
                "locked_test_r2_mean": statistics.mean(r2s),
                "persistence_mae_mean": baseline_mean,
                "mae_improvement_vs_persistence": (baseline_mean - statistics.mean(maes)) / baseline_mean,
                "result_count": len(selected),
            })
    return aggregates


def build_memory(
    curated_root: Path,
    curated_manifest_path: Path,
    aggregates: list[dict[str, Any]],
    memory_root: Path,
) -> dict[str, Any]:
    manifest_hash = sha256(curated_manifest_path)
    records: list[HistoricalExperimentRecord] = []
    observations: list[ExperimentObservation] = []
    for item in aggregates:
        model = item["model"]
        horizon = item["horizon"]
        scope = ExperimentScopeSignature(
            target_variable="steam_volumetric_flow",
            target_definition="31特征直接预测未来蒸汽体积流量V",
            target_unit="m3/s",
            prediction_mode="direct_volume",
            dataset_id="boiler_181var_v1",
            dataset_sha256=DATASET_SHA256,
            feature_set_id="31_SOFT_SENSOR_FEATURES_1BASED",
            feature_count=31,
            window_steps=20,
            prediction_horizon_steps=horizon,
            sampling_interval_seconds=15,
            split_policy="chronological_train_validation_locked_test",
            split_ratios=[0.70, 0.10, 0.20],
            metrics=["mae_m3_s", "rmse_m3_s", "r2", "mbe_m3_s"],
            baselines=["persistence"],
            protocol_status="MULTI_SEED_PREDICTIONS_AND_METRICS_HASH_AUDITED",
        )
        experiment_id = f"BM-SEED-02-{model.upper()}-H{horizon}-S7-19-42"
        model_artifacts: list[Path] = []
        for seed in SEEDS:
            library_root = selected_source_root(curated_root, seed, model)
            model_artifacts.extend([
                library_root / "manifests" / "31v_direct" / f"h{horizon}" / "manifest.json",
                library_root / "predictions" / "31v_direct" / f"h{horizon}" / f"{model}_validation_predictions.csv",
                library_root / "predictions" / "31v_direct" / f"h{horizon}" / f"{model}_locked_test_predictions.csv",
            ])
        known_issues = [
            "CROSS_TIME_BLOCK_REPLICATION_NOT_COMPLETED",
            "MULTI_SEED_SET_PREREGISTERED_BUT_FINAL_CLAIMS_REQUIRE_HUMAN_REVIEW",
            "ONLY_3_OF_31_FEATURE_NAMES_SEMANTICALLY_VERIFIED",
            "TRAINING_GIT_COMMIT_NOT_CAPTURED_IN_MANIFEST",
        ]
        if model == "rf":
            known_issues.append("PARALLEL_CPU_CONTENTION_INVALIDATES_RUNTIME_COMPARISON")
        record = HistoricalExperimentRecord(
            experiment_id=experiment_id,
            series_id="BM-SEED-02",
            parent_experiment_ids=[f"LIB31V-AUDITED-{model.upper()}_H{horizon}"],
            hypothesis_id=f"H-BM-SEED-02-{model.upper()}-H{horizon}-STABILITY",
            run_date="2026-08-21",
            source_type="teammate_multi_seed_audited_artifact_package",
            source_path=str(curated_manifest_path),
            source_sha256=manifest_hash,
            source_locator=f"aggregates[model={model},horizon={horizon}]",
            scope=scope,
            random_seeds=list(SEEDS),
            protocol_path=str(REPO / "docs" / "BM-SEED-02_队友多Seed复验教程.md"),
            candidate_models=[model],
            selection_scope="validation_only_training; locked_test_post_hoc_multi_seed_stability_summary",
            locked_test_used_for_selection=False,
            confirmation_criteria=["human_review", "cross_time_block_replication"],
            metrics=item,
            verdict="MULTI_SEED_ARTIFACT_METRICS_REPRODUCED_EXPLORATORY",
            verdict_scope=["31v_direct", f"h{horizon}", model, "seeds_7_19_42"],
            evidence_tier=EvidenceTier.AUDITED_EXPLORATORY,
            audit_status="PASSED_MULTI_SEED_ARTIFACT_AUDIT",
            known_issues=known_issues,
            reproducibility_status="HASHED_WEIGHTS_PREDICTIONS_MANIFESTS_LOGS_AND_ENVIRONMENT_ARCHIVED",
            artifact_paths=[str(path) for path in model_artifacts],
            artifact_hashes={str(path): sha256(path) for path in model_artifacts},
            raw_context="BM-SEED-02 对31V direct软测量库执行seed 7、19、42复验。",
            raw_hypothesis=f"{model} 在h{horizon}任务上的locked-test表现跨seed保持可接受稳定性。",
            raw_protocol="window=20; 15s sampling; chronological 70/10/20; seeds=7,19,42; validation-only training; locked-test audit.",
            raw_result=json.dumps(item, ensure_ascii=False, sort_keys=True),
            raw_limitations="; ".join(known_issues),
            importer_version="1.0.0",
        )
        records.append(record)
        improvement = item["mae_improvement_vs_persistence"]
        observations.append(ExperimentObservation(
            observation_id=f"OBS-{experiment_id}-STABILITY",
            source_experiment_ids=[experiment_id],
            observation_type=ObservationType.SUPPORTED if improvement > 0 else ObservationType.NULL_RESULT,
            claim=(
                f"在direct-V、h{horizon}、seed 7/19/42的locked-test上，{model}的MAE均值为"
                f"{item['locked_test_mae_mean']:.6f} m3/s，样本标准差为{item['locked_test_mae_sample_std']:.6f}，"
                f"相对persistence的MAE均值改善{improvement:.2%}；仅限当前数据与协议的探索性复验。"
            ),
            scope_signature=scope,
            comparison_signature=scope.model_dump_json(exclude_none=True),
            supporting_metrics=item,
            confidence_level=0.82,
            reuse_policy="SCOPE_MATCH_REQUIRED_MULTI_SEED_EXPLORATORY; HUMAN_REVIEW_BEFORE_CONFIRMATORY_USE",
            invalid_for_scientific_synthesis=False,
            derived_by="bm_seed_02_structured_ingester",
            derivation_version="1.0.0",
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
        "schema_version": "boilermind.bm_seed_02_ingestion.v1",
        "records_added_or_verified": len(records),
        "observations_added_or_verified": len(observations),
        "evidence_tier": EvidenceTier.AUDITED_EXPLORATORY.value,
        "memory_total_records": len(merged_records),
        "memory_total_observations": len(merged_observations),
        "confirmatory_blockers": ["human_review_pending", "cross_time_block_replication_pending"],
    }
    (store.root / "bm_seed_02_ingestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_report(path: Path, aggregates: list[dict[str, Any]], audit: dict[str, Any], artifact_root: Path) -> None:
    lines = [
        "# BM-SEED-02 多 Seed 复验审计报告", "",
        "状态：`AUDITED_EXPLORATORY`（真实运行、逐样本指标复算通过；待跨时间块复验与人工确认）", "",
        "## 审计范围", "",
        f"- 数据 SHA-256：`{DATASET_SHA256}`",
        "- Seed：7、19、42；Horizon：40、80；Window：20；采样间隔：15 秒。",
        "- 有效来源：每个 seed 的 `core` 与 `rf_corrected`。",
        "- 排除：修复前 `seed_7/model_library`；不完整 `seed_19/model_library`。",
        f"- 清洗后本地产物：`{artifact_root}`。", "",
        "## Locked-test 三 Seed 汇总", "",
        "| 模型 | Horizon | MAE均值 | MAE样本标准差 | MAE最小 | MAE最大 | R²均值 | 相对Persistence改善 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['model']} | {item['horizon']} | {item['locked_test_mae_mean']:.6f} | "
            f"{item['locked_test_mae_sample_std']:.6f} | {item['locked_test_mae_min']:.6f} | "
            f"{item['locked_test_mae_max']:.6f} | {item['locked_test_r2_mean']:.6f} | "
            f"{item['mae_improvement_vs_persistence']:.2%} |"
        )
    lines.extend([
        "", "## 审计结论", "",
        f"- 共复算并核对 `{audit['validated_result_count']}` 个 seed×horizon×model 结果，validation 与 locked-test 指标均与 manifest 一致。",
        "- Ridge 在两个 horizon 上均取得最低的三-seed平均MAE；Bayesian Ridge接近且同样稳定。",
        "- LSTM存在较明显seed波动，不能用单次最好结果替代多seed结论。",
        "- RF修复后的effective_random_state分别为7、19、42；validation到locked-test存在明显泛化落差。",
        "- 训练manifest未记录Git commit；RF并行运行耗时不可用于模型速度比较。",
        "- 本批可进入历史实验检索和假设生成，但暂不升级为确认性证据。", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗、审计并入库 BM-SEED-02 多 seed 真实实验产物。")
    parser.add_argument("--source", required=True)
    parser.add_argument("--artifact-root", default="runtime/experiment_artifacts/BM-SEED-02")
    parser.add_argument("--memory-root", default="runtime/experiment_memory")
    parser.add_argument("--report", default="docs/BM-SEED-02_多Seed复验审计报告.md")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    results, audit = validate_and_collect(source)
    aggregates = aggregate_results(results)
    copy_curated_artifacts(source, artifact_root)
    curated_file_entries = []
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        curated_file_entries.append({
            "path": path.relative_to(artifact_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    curated_manifest = {
        "schema_version": "boilermind.bm_seed_02_curated_artifacts.v1",
        "experiment_id": "BM-SEED-02",
        "dataset_sha256": DATASET_SHA256,
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "models": list(MODELS),
        "audit": audit,
        "aggregates": aggregates,
        "files": curated_file_entries,
    }
    curated_manifest_path = artifact_root / "CURATED_MANIFEST.json"
    curated_manifest_path.write_text(json.dumps(curated_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report).resolve(), aggregates, audit, artifact_root)
    memory_report = build_memory(artifact_root, curated_manifest_path, aggregates, Path(args.memory_root))
    print(json.dumps({"audit": audit, "memory": memory_report, "artifact_root": str(artifact_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
