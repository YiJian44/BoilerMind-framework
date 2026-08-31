"""e2e_data_profile.py — 数据属性驱动的软测V 单遍 E2E（AI-Scientist 单轮闭环）。

问题 → train-only 数据画像 → 6条确定性逐模型假设 H_M → 逐模型软测实验
→ 冠军池(复合判据) → 冠军 → 软测值 → 结构化报告。

符合我们设计：换假设迭代（证伪换下个模型）+ 冠军池比 MAE。
用法：PYTHONPATH=src python scripts/e2e_data_profile.py [run_id]
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "resources" / "datasets" / "boiler_181var_v1" / "boiler_181var_clean.csv"

from boilermind.experiment.capability_registry import DirectVolume31VCapabilityRegistry
from boilermind.core.contracts import ExperimentContract
from boilermind.orchestration.model_hypothesis_factory import build_model_hypotheses
from boilermind.orchestration.soft_sense import run_soft_sense_experiment
from boilermind.skills.data_profile_skill import DataProfileSkill
from boilermind.skills.profile_mapper import profile_to_model_selection


def _build_contract(family: str, *, capability, problem_id: str, hypothesis_id: str, run_id: str) -> ExperimentContract:
    snapshot = capability.snapshot()
    return ExperimentContract(
        experiment_id=f"EXP-{family.upper()}-{run_id[-6:]}",
        problem_id=problem_id,
        hypothesis_id=hypothesis_id,
        plan_id=f"PLAN-{hypothesis_id}",
        experiment_type="model_comparison",
        primary_metric="MAE",
        secondary_metrics=["RMSE", "R2", "MBE"],
        prediction_horizon_steps=0,
        sampling_interval_seconds=15,
        window_steps=20,
        locked_test_used_for_selection=False,
        required_operations=["model_comparison", "chronological_validation", "locked_test_evaluation"],
        execution_requirements={"dataset_path": str(capability.DEFAULT_DATASET_PATH)},
        dataset_id=snapshot["dataset"]["id"],
        dataset_hash="real",
        input_variables=capability.available_variables(),
        target_variable="steam_volumetric_flow",
        train_split="chronological_first_70_percent",
        validation_split="chronological_next_10_percent",
        test_split="locked_chronological_remainder",
        baseline_models=["persistence"],
        reference_models=["persistence"],
        candidate_models=[family],
        recommended_models=[family],
        executable_models=[family],
        metrics=["MAE", "RMSE", "R2", "MBE"],
        confirmation_criteria=["candidate_validation_mae_minimum", "candidate_locked_test_generalization"],
        falsification_criteria=["candidate_validation_mae_not_minimum", "candidate_locked_test_not_generalized"],
        random_seed=42,
    )


def main() -> int:
    run_id = sys.argv[1] if len(sys.argv) > 1 else f"E2E-{uuid.uuid4().hex[:8].upper()}"
    question = (
        "分析最新锅炉数据，识别数据属性（非线性、时序、稀疏化、降维、非高斯），"
        "模型库里面哪个模型软测蒸汽体积量V的误差最小"
    )
    t0 = time.time()
    capability = DirectVolume31VCapabilityRegistry(prediction_horizon_steps=0)
    out_dir = ROOT / "runtime" / "research_runs_v2" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 画像（train-only，防泄漏）
    print("== 1 数据属性画像 (train) ==", flush=True)
    profile = DataProfileSkill.compute(DATASET, horizon_steps=0, data_split="train")
    plan = profile_to_model_selection(profile, horizon_steps=0)
    print(f"  n={profile.meta.n_rows} | 候选={plan.to_run_families}", flush=True)
    print(f"  属性分: {dict(sorted(plan.property_scores.items(), key=lambda x:-x[1]))}", flush=True)

    # 2) 6 条确定性逐模型假设 H_M
    print("== 2 逐模型候选假设 ==", flush=True)
    hypotheses = build_model_hypotheses(
        profile, plan, problem_id=f"RP-{run_id[-6:]}",
    )
    print(f"  生成 {len(hypotheses)} 条: {[h['model_family'] for h in hypotheses]}", flush=True)

    # 3) 逐模型软测实验
    print("== 3 逐模型软测实验（换假设迭代 + 冠军池）==", flush=True)
    results = {}
    champion_pool = []
    for h in hypotheses:
        family = h["model_family"]
        contract = _build_contract(
            family, capability=capability,
            problem_id=h.get("problem_id") or f"RP-{run_id[-6:]}",
            hypothesis_id=h["hypothesis_id"], run_id=run_id,
        )
        print(f"  -- {family} ...", flush=True)
        out = run_soft_sense_experiment(contract, run_id=run_id)
        rec = out["experiment_result"].model_records[family]
        ss = out["scientific_result"]
        verdict = ss.verdict.value
        val = float(rec.validation_metrics["MAE"])
        locked = float(rec.locked_test_metrics["MAE"])
        in_pool = verdict == "supported"
        if in_pool:
            champion_pool.append(family)
        results[family] = {
            "hypothesis_id": h["hypothesis_id"],
            "verdict": verdict,
            "validation_mae": val,
            "locked_test_mae": locked,
            "in_champion_pool": in_pool,
            "rationale": ss.rationale,
        }
        print(f"      {family}: verdict={verdict} valMAE={val:.4f} lockMAE={locked:.4f} 池={in_pool}", flush=True)

    # 4) 冠军 = 冠军池中 validation MAE 最小
    print("== 4 冠军裁决 ==", flush=True)
    winner = min(champion_pool, key=lambda f: results[f]["validation_mae"]) if champion_pool else None
    print(f"  冠军池={champion_pool}", flush=True)
    print(f"  冠军={winner} (valMAE={results[winner]['validation_mae']:.4f})" if winner else "  冠军池空", flush=True)

    # 5) 冠军软测值
    soft_sense_values = None
    if winner:
        wc = _build_contract(
            winner, capability=capability,
            problem_id=f"RP-{run_id[-6:]}", hypothesis_id="H-CHAMPION", run_id=run_id,
        )
        wout = run_soft_sense_experiment(wc, run_id=run_id)
        soft_sense_values = wout["soft_sense_values"]

    # 6) 结构化报告
    report = {
        "schema_version": "boilermind.data_profile_e2e.v1",
        "run_id": run_id,
        "question": question,
        "baseline": "mean(无V)",
        "target_definition_id": profile.meta.target_definition_id,
        "target_formula": profile.meta.target_formula,
        "profile": {
            "n_rows": profile.meta.n_rows,
            "n_features": profile.meta.n_features,
            "v_target": profile.meta.v_target,
            "properties": {
                k: {"verdict": p.verdict, "candidate_families": p.candidate_families}
                for k, p in profile.properties.items()
            },
        },
        "hypotheses": [
            {
                "hypothesis_id": h["hypothesis_id"],
                "model": h["model_family"],
                "statement": h["hypothesis_statement"],
                "attribute_prior": h["data_attribute_prior"],
            }
            for h in hypotheses
        ],
        "model_results": results,
        "champion_pool": champion_pool,
        "winner": winner,
        "winner_validation_mae": results[winner]["validation_mae"] if winner else None,
        "winner_locked_test_mae": results[winner]["locked_test_mae"] if winner else None,
        "soft_sense_values": soft_sense_values,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 报告已写 {report_path} ==", flush=True)
    print(f"总耗时 {report['elapsed_seconds']}s | 冠军={winner} | 软测值示例={soft_sense_values}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
