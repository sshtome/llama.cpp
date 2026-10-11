@echo off
setlocal
title PopBrain Synaptic LLM CLI (Native C++)
color 0A

echo ========================================================
echo   PopBrain Synaptic LLM - Native C++ Management CLI
echo ========================================================
echo.

if exist "%~dp0popbrain_cli.exe" (
    "%~dp0popbrain_cli.exe" %*
) else if exist "%~dp0build\Release\popbrain_cli.exe" (
    "%~dp0build\Release\popbrain_cli.exe" %*
) else (
    echo [ERROR] popbrain_cli.exe not found!
    echo Please compile with CMake or download the release binary from:
    echo https://github.com/sshtome/llama.cpp/releases
    pause
    exit /b 1
)
