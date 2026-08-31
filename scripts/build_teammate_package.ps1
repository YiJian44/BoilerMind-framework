param(
  [string]$Version = 'BM-RUNNER-0.2.0',
  [string]$Destination = 'outputs\delivery'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$destinationRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Destination))
if (-not $destinationRoot.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw 'Destination 必须位于项目目录内。'
}
$packageName = "BoilerMind-Teammate-$Version"
$stagingRoot = Join-Path $destinationRoot $packageName
$zipPath = Join-Path $destinationRoot "$packageName.zip"
New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
if (Test-Path -LiteralPath $stagingRoot) { Remove-Item -LiteralPath $stagingRoot -Recurse -Force }
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null

$directories = @('src', 'scripts', 'tests', 'teammate')
foreach ($directory in $directories) {
  Copy-Item -LiteralPath (Join-Path $projectRoot $directory) -Destination (Join-Path $stagingRoot $directory) -Recurse
}
$rootFiles = @('pyproject.toml', 'requirements_frozen.txt', 'python_version.txt', '.env.example')
foreach ($file in $rootFiles) {
  Copy-Item -LiteralPath (Join-Path $projectRoot $file) -Destination (Join-Path $stagingRoot $file)
}
$dataDir = Join-Path $stagingRoot 'resources\data'
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'resources\data\shortperiod_new.csv') -Destination (Join-Path $dataDir 'shortperiod_new.csv')

Get-ChildItem -LiteralPath $stagingRoot -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $stagingRoot -Recurse -File -Include '*.pyc','*.pyo' | Remove-Item -Force
Get-ChildItem -LiteralPath $stagingRoot -Recurse -File -Filter '*.bak' | Remove-Item -Force

$gitCommit = (& git -C $projectRoot rev-parse HEAD).Trim()
$gitStatus = @(& git -C $projectRoot status --porcelain)
$manifest = [ordered]@{
  schema_version = 'boilermind.teammate_package.v1'
  package_name = $packageName
  runner_version = $Version
  source_git_commit = $gitCommit
  source_worktree_clean = ($gitStatus.Count -eq 0)
  source_contains_uncommitted_repairs = ($gitStatus.Count -gt 0)
  dataset_relative_path = 'resources/data/shortperiod_new.csv'
  dataset_sha256 = 'd52c1399b844165f94fc156fc7919be9fbb0bf214dfff74b5c48bf429917759e'
  authorized_scope = 'ENVIRONMENT_CHECK_AND_SMOKE_ONLY'
  formal_experiment_status = 'FORMAL_HOLD'
  created_at = [DateTime]::UtcNow.ToString('o')
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stagingRoot 'PACKAGE_MANIFEST.json') -Encoding utf8

$hashRows = Get-ChildItem -LiteralPath $stagingRoot -Recurse -File | ForEach-Object {
  $relative = [System.IO.Path]::GetRelativePath($stagingRoot, $_.FullName).Replace('\','/')
  [ordered]@{ path = $relative; size_bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$hashRows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stagingRoot 'SHA256SUMS.json') -Encoding utf8
Compress-Archive -LiteralPath $stagingRoot -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$delivery = [ordered]@{ zip = $zipPath; size_bytes = (Get-Item -LiteralPath $zipPath).Length; sha256 = $zipHash; staging = $stagingRoot }
$delivery | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destinationRoot "$packageName.delivery.json") -Encoding utf8
$delivery | ConvertTo-Json
