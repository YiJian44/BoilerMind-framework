$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Set-Location -LiteralPath $ProjectRoot
python -m uvicorn server.research_api.app:app --host 127.0.0.1 --port 8765
