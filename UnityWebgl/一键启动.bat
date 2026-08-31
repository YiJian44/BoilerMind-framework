@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Boiler Digital Twin Launcher

echo.
echo ========================================================
echo   Boiler Digital Twin - One Click Launcher
echo ========================================================
echo.

REM ── Try python ──
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "delims=" %%P in ('where python') do (
        echo %%P | findstr /i "WindowsApps" >nul 2>&1
        if errorlevel 1 (
            echo [OK] Python: %%P
            echo.
            "%%P" start_all.py
            goto :done
        )
    )
)

REM ── Try python3 ──
where python3 >nul 2>&1
if %errorlevel%==0 (
    for /f "delims=" %%P in ('where python3') do (
        echo %%P | findstr /i "WindowsApps" >nul 2>&1
        if errorlevel 1 (
            echo [OK] Python: %%P
            echo.
            "%%P" start_all.py
            goto :done
        )
    )
)

REM ── Try py (Windows launcher, skip WindowsApps check) ──
where py >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Python Launcher: py
    echo.
    py start_all.py
    goto :done
)

REM ── None found ──
echo [ERROR] Python not found.
echo.
echo Please install Python 3.8+ from https://www.python.org/downloads/
echo Or activate Anaconda and run: python start_all.py

:done
echo.
echo --------------------------------------------------------
if %errorlevel% neq 0 (
    echo [EXIT CODE: %errorlevel%] An error occurred.
) else (
    echo [Launcher finished]
)
echo Press any key to close this window...
pause >nul
