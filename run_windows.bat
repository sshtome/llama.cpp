@echo off
setlocal
title PopBrain Synaptic LLM Launcher
color 0A

echo ========================================================
echo   PopBrain Synaptic LLM - Zero-Backprop Local AI
echo ========================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

python "%~dp0scripts\run_synaptic_llm.py" %*

pause
