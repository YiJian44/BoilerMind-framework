from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from boilermind.core.contracts import (  # noqa: E402
    EvidenceTier, ExperimentObservation, ExperimentScopeSignature,
    HistoricalExperimentRecord, ObservationType,
)
from boilermind.experiment_memory.persistence import build_empirical_capability_profile  # noqa: E402
from boilermind.experiment_memory.store import ExperimentMemoryStore  # noqa: E402

DATASET_SHA256 = "9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c"
SEEDS = (7, 19, 42)
HORIZONS = (40, 80)
MODELS = ("persistence", "ridge", "bayesianridge", "transformer", "lstm", "rf")
REGIMES = ("overall", "steady", "ramp_up", "ramp_down", "direction_change")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    error = y_pred - y_true
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae_m3_s": float(np.mean(np.abs(error))),
        "rmse_m3_s": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator else None,
        "mbe_m3_s": float(np.mean(error)),
    }


def slopes(y_source: np.ndarray, span: int) -> tuple[np.ndarray, np.ndarray]:
    recent = np.full(len(y_source), np.nan)
    previous = np.full(len(y_source), np.nan)
    recent[span:] = (y_source[span:] - y_source[:-span]) / span
    previous[2 * span:] = (y_source[span:-span] - y_source[:-2 * span]) / span
    return recent, previous


def labels_for(recent: np.ndarray, previous: np.ndarray, indices: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.empty(len(indices), dtype=object)
    for position, index in enumerate(indices):
        current, prior = float(recent[index]), float(previous[index])
        if not np.isfinite(current) or not np.isfinite(prior):
            raise ValueError(f"insufficient_regime_history:{index}")
        if current * prior < 0 and abs(current) > threshold and abs(prior) > threshold:
            labels[position] = "direction_change"
        elif abs(current) <= threshold:
            labels[position] = "steady"
        else:
            labels[position] = "ramp_up" if current > 0 else "ramp_down"
    return labels


def prediction_path(root: Path, seed: int, horizon: int, model: str) -> Path:
    group = "rf_corrected" if model == "rf" else "core"
    return root / f"seed_{seed}" / group / "model_library" / "predictions" / "31v_direct" / f"h{horizon}" / f"{model}_locked_test_predictions.csv"


def load_prediction(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    datasets = {row["dataset_sha256"] for row in rows}
    protocols = {row["protocol_sha256"] for row in rows}
    if len(datasets) != 1 or len(protocols) != 1 or {row["split"] for row in rows} != {"locked_test"}:
        raise ValueError(f"prediction_metadata_inconsistent:{path}")
    return {
        "source": np.asarray([int(row["source_index"]) for row in rows]),
        "target": np.asarray([int(row["target_index"]) for row in rows]),
        "y_true": np.asarray([float(row["y_true"]) for row in rows]),
        "y_pred": np.asarray([float(row["y_pred"]) for row in rows]),
        "dataset_sha256": datasets.pop(), "protocol_sha256": protocols.pop(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for horizon in HORIZONS:
        for model in MODELS:
            for regime in REGIMES:
                rows = [r for r in per_seed if r["horizon_steps"] == horizon and r["model"] == model and r["regime"] == regime]
                base = [r for r in per_seed if r["horizon_steps"] == horizon and r["model"] == "persistence" and r["regime"] == regime]
                maes = [float(r["mae_m3_s"]) for r in rows]
                base_mae = statistics.mean(float(r["mae_m3_s"]) for r in base)
                r2s = [float(r["r2"]) for r in rows if r["r2"] is not None]
                output.append({
                    "experiment_id": "BM-REGIME-01", "horizon_steps": horizon, "model": model,
                    "regime": regime, "seed_count": len(rows), "sample_count_per_seed": rows[0]["sample_count"],
                    "mae_mean_m3_s": statistics.mean(maes), "mae_sample_std_m3_s": statistics.stdev(maes),
                    "mae_min_m3_s": min(maes), "mae_max_m3_s": max(maes),
                    "rmse_mean_m3_s": statistics.mean(float(r["rmse_m3_s"]) for r in rows),
                    "r2_mean": statistics.mean(r2s),
                    "mbe_mean_m3_s": statistics.mean(float(r["mbe_m3_s"]) for r in rows),
                    "mae_improvement_vs_persistence_pct": (base_mae - statistics.mean(maes)) / base_mae * 100,
                    "mae_by_seed_json": json.dumps({str(r["seed"]): r["mae_m3_s"] for r in rows}, sort_keys=True),
                })
    return output


def winners(per_seed: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for horizon in HORIZONS:
        for regime in REGIMES:
            by_seed = {}
            for seed in SEEDS:
                choices = [r for r in per_seed if r["seed"] == seed and r["horizon_steps"] == horizon and r["regime"] == regime]
                by_seed[str(seed)] = min(choices, key=lambda r: float(r["mae_m3_s"]))["model"]
            choices = [r for r in aggregates if r["horizon_steps"] == horizon and r["regime"] == regime]
            best = min(choices, key=lambda r: float(r["mae_mean_m3_s"]))
            output.append({
                "horizon_steps": horizon, "regime": regime, "mean_mae_best_model": best["model"],
                "mean_mae": best["mae_mean_m3_s"], "winner_by_seed": by_seed,
                "winner_counts": dict(Counter(by_seed.values())), "ranking_flip": len(set(by_seed.values())) > 1,
            })
    return output


def memory_ingest(out: Path, report: Path, audit_path: Path, aggregates: list[dict[str, Any]], memory_root: Path) -> dict[str, Any]:
    audit_hash = sha256(audit_path)
    records, observations = [], []
    shared_artifacts = [audit_path, report, out / "regime_metrics_by_seed.csv", out / "regime_metrics_aggregate.csv"]
    for horizon in HORIZONS:
        for model in MODELS:
            rows = [r for r in aggregates if r["horizon_steps"] == horizon and r["model"] == model]
            scope = ExperimentScopeSignature(
                target_variable="steam_volumetric_flow", target_definition="31特征直接预测未来蒸汽体积流量V",
                target_unit="m3/s", prediction_mode="direct_volume", dataset_id="boiler_181var_v1",
                dataset_sha256=DATASET_SHA256, feature_set_id="31_SOFT_SENSOR_FEATURES_1BASED",
                feature_count=31, window_steps=20, prediction_horizon_steps=horizon,
                sampling_interval_seconds=15, split_policy="chronological_train_validation_locked_test",
                split_ratios=[0.70, 0.10, 0.20],
                regime_definition="origin_only_trailing_slopes_span8_train_q75_threshold",
                metrics=["mae_m3_s", "rmse_m3_s", "r2", "mbe_m3_s"], baselines=["persistence"],
                protocol_status="MULTI_SEED_REGIME_METRICS_RECOMPUTED_FROM_HASHED_PREDICTIONS",
            )
            experiment_id = f"BM-REGIME-01-{model.upper()}-H{horizon}-S7-19-42"
            issues = ["DATA_DRIVEN_REGIME_LABELS_NOT_EXPERT_VALIDATED", "CROSS_TIME_BLOCK_REPLICATION_NOT_COMPLETED", "HUMAN_REVIEW_PENDING"]
            record = HistoricalExperimentRecord(
                experiment_id=experiment_id, series_id="BM-REGIME-01",
                parent_experiment_ids=[f"BM-SEED-02-{model.upper()}-H{horizon}-S7-19-42"],
                hypothesis_id=f"H-BM-REGIME-01-{model.upper()}-H{horizon}", run_date="2026-08-22",
                source_type="deterministic_multi_seed_regime_reanalysis", source_path=str(audit_path),
                source_sha256=audit_hash, source_locator=f"aggregate:model={model},horizon={horizon}",
                scope=scope, random_seeds=list(SEEDS), protocol_path=str(REPO / "scripts" / "run_bm_regime_01.py"),
                candidate_models=[model], selection_scope="post_hoc_regime_analysis_of_locked_test_predictions; no_retraining",
                locked_test_used_for_selection=False,
                confirmation_criteria=["expert_regime_review", "cross_time_block_replication", "human_review"],
                metrics={"overall": next(r for r in rows if r["regime"] == "overall")},
                regime_metrics={r["regime"]: r for r in rows if r["regime"] != "overall"},
                verdict="MULTI_SEED_REGIME_ANALYSIS_COMPLETED_EXPLORATORY",
                verdict_scope=["31v_direct", f"h{horizon}", model, "seeds_7_19_42", "origin_only_regimes"],
                evidence_tier=EvidenceTier.AUDITED_EXPLORATORY,
                audit_status="PASSED_MULTI_SEED_REGIME_RECOMPUTATION", known_issues=issues,
                reproducibility_status="SOURCE_PREDICTIONS_HASHED_AND_REGIME_ASSIGNMENTS_ARCHIVED",
                artifact_paths=[str(p) for p in shared_artifacts], artifact_hashes={str(p): sha256(p) for p in shared_artifacts},
                raw_context="复用BM-SEED-02三seed locked-test逐样本预测进行无未来泄漏工况分层。",
                raw_hypothesis=f"{model}在h{horizon}不同动态工况下的相对性能具有可复现边界。",
                raw_protocol="No retraining; origin-only trailing slopes; span=8; train-only q75 threshold; seeds=7,19,42.",
                raw_result=json.dumps(rows, ensure_ascii=False, sort_keys=True), raw_limitations="; ".join(issues),
                importer_version="1.0.0",
            )
            records.append(record)
            observations.append(ExperimentObservation(
                observation_id=f"OBS-{experiment_id}-REGIME-PROFILE", source_experiment_ids=[experiment_id],
                observation_type=ObservationType.BOUNDARY_CONDITION,
                claim=f"{model}在direct-volume h{horizon}的三seed工况分析已完成；仅适用于当前数据驱动工况标签。",
                scope_signature=scope, comparison_signature=scope.model_dump_json(exclude_none=True),
                supporting_metrics={r["regime"]: r for r in rows}, confidence_level=0.80,
                reuse_policy="EXACT_SCOPE_REGIME_BOUNDARY_ONLY; REQUIRE_EXPERT_AND_TIME_BLOCK_CONFIRMATION",
                invalid_for_scientific_synthesis=False, derived_by="bm_regime_01_multi_seed_ingester", derivation_version="1.0.0",
            ))
    store = ExperimentMemoryStore(memory_root)
    old_records, old_observations = store.load_records(), store.load_observations()
    record_ids, observation_ids = {r.experiment_id for r in old_records}, {o.observation_id for o in old_observations}
    merged_records = old_records + [r for r in records if r.experiment_id not in record_ids]
    merged_observations = old_observations + [o for o in observations if o.observation_id not in observation_ids]
    issues = [json.loads(line) for line in store.issues_path.read_text(encoding="utf-8").splitlines() if line.strip()] if store.issues_path.is_file() else []
    store.replace_all(merged_records, merged_observations, issues)
    (store.root / "empirical_capability_profile.json").write_text(build_empirical_capability_profile(merged_records).model_dump_json(indent=2), encoding="utf-8")
    result = {"schema_version": "boilermind.bm_regime_01_ingestion.v1", "records_added_or_verified": 12,
              "observations_added_or_verified": 12, "memory_total_records": len(merged_records),
              "memory_total_observations": len(merged_observations), "evidence_tier": "AUDITED_EXPLORATORY"}
    (store.root / "bm_regime_01_ingestion_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe multi-seed BM-REGIME-01 analysis and ingestion.")
    parser.add_argument("--artifacts", default=str(REPO / "runtime/experiment_artifacts/BM-SEED-02"))
    parser.add_argument("--cache", default=str(REPO / "runtime/31v_data"))
    parser.add_argument("--out", default=str(REPO / "runtime/experiment_artifacts/BM-REGIME-01"))
    parser.add_argument("--report", default=str(REPO / "docs/BM-REGIME-01_多Seed工况分层审计报告.md"))
    parser.add_argument("--memory-root", default=str(REPO / "runtime/experiment_memory"))
    parser.add_argument("--slope-span", type=int, default=8)
    parser.add_argument("--steady-quantile", type=float, default=0.75)
    args = parser.parse_args()
    if args.slope_span < 1 or not 0 < args.steady_quantile < 1:
        raise SystemExit("invalid_regime_parameters")
    artifacts, cache, out = Path(args.artifacts).resolve(), Path(args.cache).resolve(), Path(args.out).resolve()
    report = Path(args.report).resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_seed, horizon_audit = [], {}
    for horizon in HORIZONS:
        npz_path = cache / f"h{horizon}.npz"
        with np.load(npz_path, allow_pickle=True) as npz:
            y_source, source_indices = np.asarray(npz["y_source"], float), np.asarray(npz["source_indices"], int)
            target_indices, train_indices = np.asarray(npz["target_indices"], int), np.asarray(npz["train_idx"], int)
            locked_indices, expected_y = np.asarray(npz["locked_test_idx"], int), np.asarray(npz["y"], float)[np.asarray(npz["locked_test_idx"], int)]
        recent, previous = slopes(y_source, args.slope_span)
        valid_train = train_indices[train_indices >= 2 * args.slope_span]
        threshold = float(np.quantile(np.abs(recent[valid_train]), args.steady_quantile))
        regime_labels = labels_for(recent, previous, locked_indices, threshold)
        assignment_path = out / f"h{horizon}_regime_assignments.csv"
        write_csv(assignment_path, [{"source_index": int(source_indices[i]), "target_index": int(target_indices[i]),
                                     "recent_slope_m3_s_per_step": float(recent[i]), "previous_slope_m3_s_per_step": float(previous[i]),
                                     "regime": str(label)} for i, label in zip(locked_indices, regime_labels)])
        protocols = set()
        for seed in SEEDS:
            loaded = {}
            for model in MODELS:
                item = load_prediction(prediction_path(artifacts, seed, horizon, model))
                if item["dataset_sha256"] != DATASET_SHA256 or not np.array_equal(item["source"], source_indices[locked_indices]) or not np.array_equal(item["target"], target_indices[locked_indices]) or not np.allclose(item["y_true"], expected_y, rtol=0, atol=5e-7):
                    raise ValueError(f"prediction_alignment_failed:{seed}:{horizon}:{model}")
                loaded[model] = item
                protocols.add(item["protocol_sha256"])
            base = {}
            for regime in REGIMES:
                mask = np.ones(len(regime_labels), bool) if regime == "overall" else regime_labels == regime
                base[regime] = metrics(loaded["persistence"]["y_true"][mask], loaded["persistence"]["y_pred"][mask])
            for model, item in loaded.items():
                for regime in REGIMES:
                    mask = np.ones(len(regime_labels), bool) if regime == "overall" else regime_labels == regime
                    current = metrics(item["y_true"][mask], item["y_pred"][mask])
                    base_mae = float(base[regime]["mae_m3_s"])
                    per_seed.append({"experiment_id": "BM-REGIME-01", "seed": seed, "horizon_steps": horizon,
                                     "model": model, "regime": regime, "sample_count": int(np.sum(mask)), **current,
                                     "mae_improvement_vs_persistence_pct": (base_mae - float(current["mae_m3_s"])) / base_mae * 100,
                                     "dataset_sha256": item["dataset_sha256"], "protocol_sha256": item["protocol_sha256"]})
        counts = Counter(regime_labels.tolist())
        horizon_audit[f"h{horizon}"] = {"cache_sha256": sha256(npz_path), "locked_test_count": len(locked_indices),
            "steady_threshold_m3_s_per_step": threshold, "regime_counts": {r: counts.get(r, 0) for r in REGIMES[1:]},
            "protocol_sha256_values": sorted(protocols), "assignment_sha256": sha256(assignment_path)}
    aggregates = aggregate(per_seed)
    winner_rows = winners(per_seed, aggregates)
    by_seed_path, aggregate_path = out / "regime_metrics_by_seed.csv", out / "regime_metrics_aggregate.csv"
    write_csv(by_seed_path, per_seed)
    write_csv(aggregate_path, aggregates)
    audit = {"schema_version": "boilermind.bm_regime_01.v2", "experiment_id": "BM-REGIME-01",
             "status": "COMPLETED_MULTI_SEED_EXPLORATORY", "seeds": list(SEEDS), "models": list(MODELS),
             "leakage_control": "regimes_use_y_source_at_or_before_prediction_origin_only",
             "slope_span_steps": args.slope_span, "steady_threshold_quantile_train_only": args.steady_quantile,
             "horizons": horizon_audit, "per_seed_row_count": len(per_seed), "aggregate_row_count": len(aggregates),
             "per_seed_sha256": sha256(by_seed_path), "aggregate_sha256": sha256(aggregate_path), "winner_summary": winner_rows}
    audit_path = out / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# BM-REGIME-01 多 Seed 工况分层审计报告", "", "状态：`AUDITED_EXPLORATORY`。复用BM-SEED-02的seed 7、19、42逐样本预测，不重新训练模型。", "",
             "工况只使用预测起点及更早的真实V序列；稳态阈值只由训练段计算。", "", "## 三 Seed 工况最佳模型", "",
             "| Horizon | Regime | 三Seed平均MAE最佳 | MAE均值 | Seed获胜模型 | 排名翻转 |", "|---|---|---|---:|---|---|"]
    for row in winner_rows:
        seed_text = ", ".join(f"{seed}:{model}" for seed, model in row["winner_by_seed"].items())
        lines.append(f"| h{row['horizon_steps']} | {row['regime']} | {row['mean_mae_best_model']} | {float(row['mean_mae']):.6f} | {seed_text} | {'是' if row['ranking_flip'] else '否'} |")
    lines += ["", "## 证据边界", "", "- 工况标签是数据驱动定义，不等同于锅炉专家正式工况标签。", "- 三seed复验不能替代跨时间块复验。", "- 单个seed或工况的最优不能直接确定部署模型。", "- 结论进入历史实验库，但仍需人工批准才能升级为确认性证据。", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    memory = memory_ingest(out, report, audit_path, aggregates, Path(args.memory_root).resolve())
    print(json.dumps({"audit": audit, "memory": memory}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
