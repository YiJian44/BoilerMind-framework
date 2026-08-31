$ErrorActionPreference = 'Stop'
$desk = [Environment]::GetFolderPath('Desktop')
$root = Join-Path $desk 'BoilerMind_h40_research'
$stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$pkg = Join-Path $root "build_$stamp"
New-Item -ItemType Directory -Force -Path $pkg | Out-Null
$q1 = Join-Path $pkg 'Q1_DATA_PROFILE_BEST'
$q2 = Join-Path $pkg 'Q2_H40_MODEL_COMPARE'
New-Item -ItemType Directory -Force -Path $q1, $q2 | Out-Null

$src = 'E:\AI-Workspace\30_Projects\active\BoilerMind 正式版有科研假设的更新迭代'

Copy-Item -Path (Join-Path $src 'runtime\research_runs_v2\Q1-DATA-PROFILE-BEST\*') -Destination $q1 -Recurse -Force
Copy-Item -Path (Join-Path $src 'runtime\stdout_Q1-DATA-PROFILE-BEST.log') -Destination $q1 -Force

Copy-Item -Path (Join-Path $src 'runtime\research_runs_v2\Q2-H40-MODEL-COMPARE\*') -Destination $q2 -Recurse -Force
Copy-Item -Path (Join-Path $src 'outputs\experiments\Q2-FOUR-MODEL-H40') -Destination (Join-Path $q2 'direct_backend_supplement') -Recurse -Force
Copy-Item -Path (Join-Path $src 'runtime\stdout_Q2-H40-MODEL-COMPARE.log') -Destination $q2 -Force

$readme = "BoilerMind real-pipeline outputs`n"
$readme += "================================`n"
$readme += "Generated : $stamp`n"
$readme += "Source    : $src`n"
$readme += "Pipeline  : scripts/run_full_e2e.py (ResearchOrchestrator)`n"
$readme += "Env       : conda pytorch_env (Python 3.11.14), Qwen DashScope compatible-mode`n`n"
$readme += "Q1_DATA_PROFILE_BEST`n"
$readme += "--------------------`n"
$readme += "Run ID    : Q1-DATA-PROFILE-BEST`n"
$readme += "Question  : Analyse boiler data-property profile; identify the lowest-error / highest-soft-sense-accuracy model.`n"
$readme += "Hypotheses: H001 ridge, H002 bayesianridge, H003 pls, H004 lstm, H005 gru, H006 hgb (Qwen-generated).`n"
$readme += "Status    : COMPLETED (run.json)`n"
$readme += "Best by locked-test MAE (full pipeline, 181V dataset): **PLS** MAE=0.1372 t/h, R^2=+0.129`n"
$readme += "Files:`n"
$readme += "  data_profile.json            Data-property profile`n"
$readme += "  model_selection.json         Profile-derived model plan`n"
$readme += "  run.json                     Full state machine`n"
$readme += "  structured_report.json       Structured outcome`n"
$readme += "  narrative_report.md          Narrative outcome`n"
$readme += "  scientific_research_plan/    Generated plan (.md/.docx/.json/.manifest.json)`n"
$readme += "  stdout_Q1-DATA-PROFILE-BEST.log   Process stdout`n`n"
$readme += "Q2_H40_MODEL_COMPARE`n"
$readme += "--------------------`n"
$readme += "Run ID    : Q2-H40-MODEL-COMPARE`n"
$readme += "Question  : Compare Ridge, BayesianRidge, RandomForest vs Persistence on h40 (10-min-ahead) steam-volumetric forecast.`n"
$readme += "Pipeline run (LLM-driven, 181V dataset): pls < hgb < ridge < bayesianridge < gru < lstm.`n"
$readme += "Qwen did not include RandomForest in the default candidates; the direct_backend_supplement/`n"
$readme += "run fills that gap using RealSklearnExperimentBackend on resources/data/shortperiod_new.csv`n"
$readme += "(31-column h40 short-period dataset, matching the backend contract). 4-way locked-test (t/h):`n"
$readme += "  persistence 28.85 < bayesianridge 29.11 < ridge 30.22 < rf 47.88`n"
$readme += "Best in the strict 4-way: **Persistence** baseline (BayesianRidge within 0.27 t/h; RandomForest`n"
$readme += "falls far behind due to high-lag windowed flatten and strong self-correlation).`n"
$readme += "Files:`n"
$readme += "  run.json                     Pipeline state machine`n"
$readme += "  structured_report.json       Pipeline structured outcome`n"
$readme += "  narrative_report.md          Pipeline narrative outcome`n"
$readme += "  scientific_research_plan/    Pipeline-generated plan`n"
$readme += "  direct_backend_supplement/   Direct-backend 4-model run (incl. rf + persistence)`n"
$readme += "    experiment_result.json     Full metrics`n"
$readme += "    {ridge,bayesianridge,rf}.joblib`n"
$readme += "    {ridge,bayesianridge,rf}_locked_test_predictions.csv`n"
$readme += "  stdout_Q2-H40-MODEL-COMPARE.log   Process stdout`n`n"
$readme += "Known non-fatal warnings (both runs):`n"
$readme += "- literature_retrieval FAILED: httpx vs httpx2 client mismatch (does NOT block pipeline).`n"
$readme += "- evolution_sink warning: experiment_result.experiment_valid must be a boolean.`n"

$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $pkg 'README.txt'), $readme, $utf8)

Write-Host ""
Write-Host "Package : $pkg"
Write-Host ""
$size = (Get-ChildItem $pkg -Recurse | Where-Object { -not $_.PSIsContainer } | Measure-Object -Property Length -Sum).Sum
Write-Host ("Size    : {0:N1} KB ({1:N2} MB)" -f ($size/1KB), ($size/1MB))
Write-Host ""
Write-Host "Q1 files:"
Get-ChildItem $q1 -Recurse | Where-Object { -not $_.PSIsContainer } | Select-Object FullName | ForEach-Object { "  $($_.FullName.Substring($q1.Length))" }
Write-Host ""
Write-Host "Q2 files:"
Get-ChildItem $q2 -Recurse | Where-Object { -not $_.PSIsContainer } | Select-Object FullName | ForEach-Object { "  $($_.FullName.Substring($q2.Length))" }
