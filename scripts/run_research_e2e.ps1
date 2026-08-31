# BoilerMind real-pipeline launcher.
# pytorch_env (Python 3.11.14) + .env.local Qwen DashScope-compatible API.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Question,
    [Parameter(Mandatory = $true)][string]$RunId,
    [int]$TimeoutMinutes = 30
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe   = "E:\conda_envs\pytorch_env\python.exe"
if (-not (Test-Path $PythonExe)) { throw "missing $PythonExe" }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:BOILERMIND_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:BOILERMIND_QWEN_MODEL    = "qwen3.7-plus"
$env:BOILERMIND_QWEN_TIMEOUT  = "240"
$env:BOILERMIND_QWEN_MAX_TOKENS = "4096"
$env:BOILERMIND_ENABLE_WEB_LITERATURE = "0"
$env:BOILERMIND_REAL_DATASET_PATH = Join-Path $ProjectRoot "resources\data\shortperiod_new.csv"

$envLocal = Join-Path $ProjectRoot ".env.local"
if (Test-Path $envLocal) {
    $line = Select-String -Path $envLocal -Pattern '^DASHSCOPE_API_KEY=(.+)$' | Select-Object -First 1
    if ($line) {
        $env:DASHSCOPE_API_KEY = $line.Matches[0].Groups[1].Value
    }
}
if (-not $env:DASHSCOPE_API_KEY) { throw "DASHSCOPE_API_KEY missing in .env.local" }

Write-Host "[run] ProjectRoot: $ProjectRoot"
Write-Host "[run] Python    : $PythonExe"
Write-Host "[run] Dataset   : $($env:BOILERMIND_REAL_DATASET_PATH)"
Write-Host "[run] Qwen base : $($env:BOILERMIND_QWEN_BASE_URL)"
Write-Host "[run] Qwen model: $($env:BOILERMIND_QWEN_MODEL)"
Write-Host "[run] Run ID    : $RunId"
Write-Host "[run] Question  : $Question"
Write-Host "[run] Timeout   : $TimeoutMinutes min"
Write-Host ""

Set-Location $ProjectRoot
$env:PYTHONPATH = (Resolve-Path (Join-Path $ProjectRoot "src")).Path

$stdoutLog = Join-Path $ProjectRoot "runtime\stdout_${RunId}.log"
$stderrLog = Join-Path $ProjectRoot "runtime\stderr_${RunId}.log"
New-Item -ItemType Directory -Force -Path (Split-Path $stdoutLog) | Out-Null

$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList @("-u", "scripts\run_full_e2e.py", "--question", $Question, "--run-id", $RunId) `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError  $stderrLog

$timeoutSec = $TimeoutMinutes * 60
$exited = $proc.WaitForExit($timeoutSec)
if (-not $exited) {
    Write-Warning "TIMEOUT after $TimeoutMinutes min; killing PID=$($proc.Id)"
    try { Stop-Process -Id $proc.Id -Force -ErrorAction Stop } catch {}
    $exit = 124
} else {
    $exit = $proc.ExitCode
}

Write-Host ""
Write-Host "--- stdout (last 120 lines) ---"
if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Tail 120 }
Write-Host "--- stderr (last 60 lines) ---"
if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 60 }

$runArtifact = Join-Path $ProjectRoot "runtime\research_runs_v2\$RunId\run.json"
if (Test-Path $runArtifact) {
    Write-Host "[run] artifact: $runArtifact"
} else {
    Write-Warning "[run] no run.json produced"
}

exit $exit
