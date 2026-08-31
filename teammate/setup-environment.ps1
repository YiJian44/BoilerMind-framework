param([switch]$SkipTorch)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venvPath = Join-Path $projectRoot '.venv'
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw '未找到 py.exe，请先安装 Python 3.11 x64。' }
& py -3.11 -m venv $venvPath
if ($LASTEXITCODE -ne 0) { throw '创建 Python 3.11 虚拟环境失败。' }
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw '升级 pip 失败。' }
& $pythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements-teammate.txt')
if ($LASTEXITCODE -ne 0) { throw '安装队友任务依赖失败。' }
if (-not $SkipTorch) {
  & $pythonExe -m pip install torch
  if ($LASTEXITCODE -ne 0) { throw '安装 Torch 失败。' }
}
& $pythonExe -m pip install -e $projectRoot
if ($LASTEXITCODE -ne 0) { throw '安装 BoilerMind 项目失败。' }
Write-Host "环境安装完成：$venvPath"
