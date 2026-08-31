# DEPRECATED: 脚本保留仅为回退 / 队友环境兼容。新代码请用:
#   python scripts/run_stack.py backend
# 跨平台、可测、选择 venv 顺序同 run_stack.py。

param([int]$Port = 8765)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$environmentScript = Join-Path $projectRoot 'handoff_env.ps1'

if (Test-Path -LiteralPath $environmentScript) {
    . $environmentScript
}

# 自动加载 .env.local（不打印任何值，避免泄露密钥）
$envFile = Join-Path $projectRoot '.env.local'
if (Test-Path -LiteralPath $envFile) {
    Get-Content -LiteralPath $envFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $eq = $line.IndexOf('=')
            $name = $line.Substring(0, $eq).Trim()
            $value = $line.Substring($eq + 1).Trim()
            $value = $value.Trim('"').Trim("'")
            if ($name -match '^[A-Za-z_][A-Za-z0-9_]*$') {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
}

# 优先使用项目内 .venv（与 requirements-frozen / pyproject 一致），
# 其次回退到历史路径 D:\anaconda\envs\boilermind311，最后回退到系统 python。
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$condaPython = 'D:\anaconda\envs\boilermind311\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} elseif (Test-Path -LiteralPath $condaPython) {
    $python = $condaPython
} else {
    $python = 'python'
}

Set-Location -LiteralPath $projectRoot
Write-Host "BoilerMind backend starting at http://127.0.0.1:$Port (python: $python)" -ForegroundColor Green
& $python -m uvicorn server.research_api.app:app --host 127.0.0.1 --port $Port
