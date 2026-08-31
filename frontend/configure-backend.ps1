param(
  [Parameter(Mandatory = $true)]
  [string]$BackendUrl
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $projectRoot 'js\config.js'
$normalizedUrl = $BackendUrl.Trim().TrimEnd('/')

if ($normalizedUrl -notmatch '^https?://[^/]+(?::\d+)?$') {
  throw '后端地址格式不正确。示例：http://127.0.0.1:8765 或 http://192.168.1.20:8765'
}

$content = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8
$pattern = "apiBaseUrl:\s*'[^']*'"
if ($content -notmatch $pattern) {
  throw '未能找到 apiBaseUrl 配置项，请确认交付包文件完整。'
}
$updated = $content -replace $pattern, "apiBaseUrl: '$normalizedUrl'"

[System.IO.File]::WriteAllText($configPath, $updated, [System.Text.UTF8Encoding]::new($false))
Write-Host "后端地址已设置为：$normalizedUrl" -ForegroundColor Green
Write-Host '如果前端已打开，请按 Ctrl+F5 强制刷新页面。'
