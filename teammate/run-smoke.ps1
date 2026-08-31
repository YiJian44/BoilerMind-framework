param(
  [Parameter(Mandatory=$true)][ValidateSet('BM-SMOKE-02-RIDGE-S42','BM-SMOKE-02-DLINEAR-S42')][string]$TaskId
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$taskFile = Join-Path $PSScriptRoot 'tasks\smoke_tasks.json'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '未找到 .venv，请先配置并检查环境。' }
& $pythonExe (Join-Path $projectRoot 'scripts\run_teammate_task_with_timeout.py') --task-file $taskFile --task-id $TaskId
exit $LASTEXITCODE
