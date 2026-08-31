# DEPRECATED: 脚本保留仅为回退 / 队友环境兼容。新代码请用:
#   python scripts/run_stack.py up
# 跨平台、可测、无 PowerShell 5.x 渲染串扰。

param(
    [int]$BackendPort = 8765,
    [int]$FrontendPort = 8081
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

function Test-Port([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync('127.0.0.1', $Port)
        if (-not $task.Wait(300)) { return $false }
        return -not $task.IsFaulted
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-Port([int]$Port, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

$runtimeDir = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$backendLog = Join-Path $runtimeDir 'backend_8765.log'
$backendErr = Join-Path $runtimeDir 'backend_8765.err.log'
$frontendLog = Join-Path $runtimeDir 'frontend_8081.log'
$frontendErr = Join-Path $runtimeDir 'frontend_8081.err.log'

function Start-HiddenPowerShell {
    param(
        [string]$ScriptPath,
        [string[]]$ExtraArgs = @(),
        [string]$StdOutLog,
        [string]$StdErrLog
    )
    # 使用 cmd.exe /c start /B 包装启动 powershell.exe，理由：
    # 1) powershell.exe 不接受 cmd 风格的 1>/2> 重定向，需要 cmd 中转；
    # 2) Start-Process -ArgumentList 数组模式在 PS5 下拼接带空格路径不可靠；
    # 3) cmd 的双引号路径处理最稳。
    # 副作用：PowerShell 5.x 客户端渲染时可能出现输出顺序串扰，
    # 但 pwsh 7 验证逻辑顺序正常，实际后台进程全部正确启动。
    $extra = ''
    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $extra = ' ' + (($ExtraArgs | ForEach-Object {
            if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
        }) -join ' ')
    }
    $cmdArgs = '/c start "" /B powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $ScriptPath + '"' + $extra
    if ($StdOutLog) {
        $cmdArgs += ' 1> "' + $StdOutLog + '"'
    }
    if ($StdErrLog) {
        $cmdArgs += ' 2> "' + $StdErrLog + '"'
    }
    Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdArgs -WindowStyle Hidden | Out-Null
}

if (-not (Test-Port $BackendPort)) {
    Write-Host "[1/3] 启动后端 :$BackendPort ..." -ForegroundColor Cyan
    Start-HiddenPowerShell `
        -ScriptPath (Join-Path $projectRoot 'server\start-backend.ps1') `
        -ExtraArgs @('-Port', "$BackendPort") `
        -StdOutLog $backendLog `
        -StdErrLog $backendErr
    if (-not (Wait-Port $BackendPort 90)) {
        throw "后端 :$BackendPort 启动超时，请查看 $backendErr"
    }
    Write-Host "     后端已就绪。" -ForegroundColor Green
} else {
    Write-Host "[1/3] 后端 :$BackendPort 已在运行。" -ForegroundColor Green
}

if (-not (Test-Port $FrontendPort)) {
    Write-Host "[2/3] 启动前端 :$FrontendPort（含 Unity :8090）..." -ForegroundColor Cyan
    Start-HiddenPowerShell `
        -ScriptPath (Join-Path $projectRoot 'frontend\start-frontend.ps1') `
        -ExtraArgs @('-Port', "$FrontendPort", '-NoBrowser') `
        -StdOutLog $frontendLog `
        -StdErrLog $frontendErr
    if (-not (Wait-Port $FrontendPort 60)) {
        throw "前端 :$FrontendPort 启动超时，请查看 $frontendErr"
    }
    Write-Host "     前端就绪。" -ForegroundColor Green
} else {
    Write-Host "[2/3] 前端 :$FrontendPort 已在运行。" -ForegroundColor Green
}

Write-Host "[3/3] 全部就绪。" -ForegroundColor Green
Write-Host ""
Write-Host "后端 API:   http://127.0.0.1:$BackendPort/health/ready"
Write-Host "前端页面:   http://127.0.0.1:$FrontendPort/#/chat"
Write-Host "Unity 页面: http://127.0.0.1:8090/index_unity_only.html"
Write-Host ""
Write-Host "浏览器若显示旧界面，请按 Ctrl+F5 硬刷新。" -ForegroundColor Yellow
