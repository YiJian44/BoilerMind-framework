$ProjectRoot = $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$env:BOILERMIND_QWEN_BASE_URL = "https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:BOILERMIND_QWEN_MODEL = "qwen3.7-plus"

$env:BOILERMIND_ENABLE_WEB_LITERATURE = "0"

$env:BOILERMIND_REAL_DATASET_PATH = Join-Path `
    $ProjectRoot `
    "resources\data\shortperiod_new.csv"

Write-Host "BoilerMind handoff environment loaded."
Write-Host "Dataset:"
Write-Host $env:BOILERMIND_REAL_DATASET_PATH

Write-Host ""
Write-Host "IMPORTANT:"
Write-Host "Set DASHSCOPE_API_KEY separately before running."
