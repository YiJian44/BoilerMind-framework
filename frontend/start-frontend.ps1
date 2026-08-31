# DEPRECATED: 脚本保留仅为回退 / 队友环境兼容。新代码请用:
#   python scripts/run_stack.py frontend
# 跨平台、可测。

param(
    [int]$Port = 8080,
    [string]$UnityRoot = '',
    [int]$UnityPort = 8090,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$frontendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $UnityRoot) {
    $bundledUnityRoot = Join-Path (Split-Path -Parent $frontendRoot) 'UnityWebgl'
    if (Test-Path -LiteralPath (Join-Path $bundledUnityRoot 'Build\UnityWebgl.loader.js')) {
        $UnityRoot = $bundledUnityRoot
    }
}

function Get-PythonCommand {
    $projectRoot = Split-Path -Parent $frontendRoot
    $candidates = @(
        @{ FilePath = (Join-Path $projectRoot '.venv\Scripts\python.exe'); Prefix = @() },
        @{ FilePath = 'D:\anaconda\envs\boilermind311\python.exe'; Prefix = @() },
        @{ FilePath = 'python'; Prefix = @() },
        @{ FilePath = 'py'; Prefix = @('-3') }
    )

    foreach ($candidate in $candidates) {
        try {
            if (($candidate.FilePath -match '^[A-Za-z]:\\') -and
                (-not (Test-Path -LiteralPath $candidate.FilePath))) {
                continue
            }
            & $candidate.FilePath @($candidate.Prefix) --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
            continue
        }
    }
    throw 'Python 3 was not found. Activate the BoilerMind environment and try again.'
}

function Test-LocalPort {
    param([int]$LocalPort)
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connectTask = $client.ConnectAsync('127.0.0.1', $LocalPort)
        if (-not $connectTask.Wait(250)) {
            return $false
        }
        if ($connectTask.IsFaulted) {
            return $false
        }
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Start-PythonProcess {
    param(
        [hashtable]$PythonCommand,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )
    $processArguments = foreach ($argument in (@($PythonCommand.Prefix) + $Arguments)) {
        if ($argument -match '[\s"]') {
            '"' + ($argument -replace '"', '\"') + '"'
        } else {
            $argument
        }
    }
    return Start-Process -FilePath $PythonCommand.FilePath `
        -ArgumentList $processArguments `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -PassThru
}

$python = Get-PythonCommand
$frontendProcess = $null
$requestedPort = $Port

# Do not mistake an unrelated local service for the BoilerMind frontend.
# Search a small, predictable range when the requested port is occupied.
while (Test-LocalPort -LocalPort $Port) {
    $Port++
    if ($Port -gt ($requestedPort + 20)) {
        throw "No free frontend port was found between $requestedPort and $($requestedPort + 20)."
    }
}

if ($Port -ne $requestedPort) {
    Write-Warning "Port $requestedPort is occupied by another service. Using port $Port instead."
}

$serverScript = Join-Path $frontendRoot 'serve_frontend.py'
if (-not (Test-Path -LiteralPath $serverScript)) {
    throw "Frontend server script was not found: $serverScript"
}
$frontendProcess = Start-PythonProcess -PythonCommand $python -Arguments @(
    $serverScript, '--port', [string]$Port, '--directory', $frontendRoot
) -WorkingDirectory $frontendRoot
foreach ($attempt in 1..50) {
    if (Test-LocalPort -LocalPort $Port) { break }
    Start-Sleep -Milliseconds 100
}

if (-not (Test-LocalPort -LocalPort $Port)) {
    throw "Frontend failed to start: port $Port is not listening."
}

if ($UnityRoot) {
    $unityLoader = Join-Path $UnityRoot 'Build\UnityWebgl.loader.js'
    if (-not (Test-Path -LiteralPath $unityLoader)) {
        Write-Warning "Unity was skipped because the loader was not found: $unityLoader"
    } elseif (-not (Test-LocalPort -LocalPort $UnityPort)) {
        [void](Start-PythonProcess -PythonCommand $python -Arguments @(
            '-m', 'http.server', [string]$UnityPort, '--directory', $UnityRoot
        ) -WorkingDirectory $UnityRoot)
    }
}

$frontendUrl = "http://127.0.0.1:$Port/#/chat"
Write-Host "BoilerMind frontend started: $frontendUrl" -ForegroundColor Green
if ($frontendProcess) {
    Write-Host "Frontend process ID: $($frontendProcess.Id)"
}
Write-Host 'Closing this PowerShell window will not stop the server process.'

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
