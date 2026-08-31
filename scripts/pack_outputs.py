"""Pack the two research runs to a timestamped folder on the Desktop."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

SRC = Path(r"E:\AI-Workspace\30_Projects\active\BoilerMind 正式版有科研假设的更新迭代")
DESK = Path.home() / "Desktop"
ROOT = DESK / "BoilerMind_h40_research"

stamp = time.strftime("%Y%m%d_%H%M%S")
pkg = ROOT / f"build_{stamp}"
q1 = pkg / "Q1_DATA_PROFILE_BEST"
q2 = pkg / "Q2_H40_MODEL_COMPARE"

pkg.mkdir(parents=True, exist_ok=True)
q1.mkdir(parents=True, exist_ok=True)
q2.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  WARN missing {src}")
        return
    if src.is_file():
        shutil.copy2(src, dst / src.name)
        return
    shutil.copytree(src, dst / src.name, dirs_exist_ok=True)


def file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


# Q1
copy_tree(SRC / "runtime/research_runs_v2/Q1-DATA-PROFILE-BEST", q1)
shutil.copy2(SRC / "runtime/stdout_Q1-DATA-PROFILE-BEST.log", q1)

# Q2 (pipeline run)
copy_tree(SRC / "runtime/research_runs_v2/Q2-H40-MODEL-COMPARE", q2)
# Q2 (direct-backend supplement with rf)
copy_tree(SRC / "outputs/experiments/Q2-FOUR-MODEL-H40", q2 / "direct_backend_supplement")
shutil.copy2(SRC / "runtime/stdout_Q2-H40-MODEL-COMPARE.log", q2)

# README
readme = f"""BoilerMind real-pipeline outputs
================================
Generated : {stamp}
Source    : {SRC}
Pipeline  : scripts/run_full_e2e.py (ResearchOrchestrator)
Env       : conda pytorch_env (Python 3.11.14), Qwen DashScope compatible-mode

Q1_DATA_PROFILE_BEST
--------------------
Run ID    : Q1-DATA-PROFILE-BEST
Question  : Analyse boiler data-property profile; identify the lowest-error / highest-soft-sense-accuracy model.
Hypotheses: H001 ridge, H002 bayesianridge, H003 pls, H004 lstm, H005 gru, H006 hgb (Qwen-generated).
Status    : COMPLETED (run.json)
Best by locked-test MAE (full pipeline, 181V dataset): **PLS** MAE=0.1372 t/h, R^2=+0.129
Files:
  data_profile.json            Data-property profile
  model_selection.json         Profile-derived model plan
  run.json                     Full state machine
  structured_report.json       Structured outcome
  narrative_report.md          Narrative outcome
  scientific_research_plan/    Generated plan (.md/.docx/.json/.manifest.json)
  stdout_Q1-DATA-PROFILE-BEST.log   Process stdout

Q2_H40_MODEL_COMPARE
--------------------
Run ID    : Q2-H40-MODEL-COMPARE
Question  : Compare Ridge, BayesianRidge, RandomForest vs Persistence on h40 (10-min-ahead) steam-volumetric forecast.
Pipeline run (LLM-driven, 181V dataset): pls < hgb < ridge < bayesianridge < gru < lstm.
Qwen did not include RandomForest in the default candidates; the direct_backend_supplement/
run fills that gap using RealSklearnExperimentBackend on resources/data/shortperiod_new.csv
(31-column h40 short-period dataset, matching the backend contract). 4-way locked-test (t/h):
  persistence 28.85 < bayesianridge 29.11 < ridge 30.22 < rf 47.88
Best in the strict 4-way: **Persistence** baseline (BayesianRidge within 0.27 t/h; RandomForest
falls far behind due to high-lag windowed flatten and strong self-correlation).

Files:
  run.json                     Pipeline state machine
  structured_report.json       Pipeline structured outcome
  narrative_report.md          Pipeline narrative outcome
  scientific_research_plan/    Pipeline-generated plan
  direct_backend_supplement/   Direct-backend 4-model run (incl. rf + persistence)
    experiment_result.json     Full metrics
    {{ridge,bayesianridge,rf}}.joblib
    {{ridge,bayesianridge,rf}}_locked_test_predictions.csv
  stdout_Q2-H40-MODEL-COMPARE.log   Process stdout

Known non-fatal warnings (both runs):
- literature_retrieval FAILED: httpx vs httpx2 client mismatch (does NOT block pipeline).
- evolution_sink warning: experiment_result.experiment_valid must be a boolean.
"""
(pkg / "README.txt").write_text(readme, encoding="utf-8")


def list_dir(p: Path) -> list[str]:
    files: list[str] = []
    for f in sorted(p.rglob("*")):
        if f.is_file():
            files.append(str(f.relative_to(p)))
    return files


total = 0
for f in (q1.rglob("*"), q2.rglob("*")):
    pass


def total_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


print()
print(f"Package : {pkg}")
print(f"Size    : {total_size(pkg)/1024:.1f} KB ({total_size(pkg)/1024/1024:.2f} MB)")
print()
print("Q1 files:")
for rel in list_dir(q1):
    print(f"  {rel}")
print()
print("Q2 files:")
for rel in list_dir(q2):
    print(f"  {rel}")
print()
print(f"README  : {pkg / 'README.txt'}")
