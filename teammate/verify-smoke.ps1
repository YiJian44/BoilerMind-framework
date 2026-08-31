param(
  [Parameter(Mandatory=$true)][ValidateSet('ridge','dlinear')][string]$Model
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$taskId = if ($Model -eq 'ridge') { 'BM-SMOKE-02-RIDGE-S42' } else { 'BM-SMOKE-02-DLINEAR-S42' }
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '未找到 .venv，请先配置并检查环境。' }
& $pythonExe (Join-Path $projectRoot 'scripts\verify_repair_smoke.py') (Join-Path $projectRoot "outputs\teammate\model_runs\$taskId") --model $Model
exit $LASTEXITCODE
