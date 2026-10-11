@echo off
setlocal
echo ========================================================================
echo  PopBrain ANN-to-SNN Neuromorphic Model Converter
echo ========================================================================
echo Input Model:    D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf
echo Output Model:   D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf
echo LIF Threshold:  0.04
echo Membrane Tau:   2.0
echo Target Sparsity:75%%
echo ========================================================================

cd /d "%~dp0"
..\build\bin\Release\popbrain_cli.exe --convert-to-spiking D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf --threshold 0.04 --tau 2.0

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [+] Conversion to Spiking SNN completed successfully!
) else (
    echo.
    echo [ERROR] Conversion failed with code %ERRORLEVEL%
)

pause
