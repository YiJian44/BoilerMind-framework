param(
  [Parameter(Mandatory=$true)][string]$OperatorId,
  [Parameter(Mandatory=$true)][string]$MachineId
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '未找到 .venv，请先运行 setup-environment.ps1。' }
& $pythonExe (Join-Path $projectRoot 'scripts\verify_teammate_environment.py') --operator-id $OperatorId --machine-id $MachineId
exit $LASTEXITCODE
