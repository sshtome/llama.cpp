@echo off
setlocal
echo ========================================================================
echo  PopBrain SNN Event-Driven Spike Dynamics Benchmark
echo ========================================================================
echo Model:         D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf
echo LIF Threshold: 0.04
echo Membrane Tau:  2.0
echo Steps:         1000 tokens
echo ========================================================================

cd /d "%~dp0"
..\build\bin\Release\popbrain_cli.exe --spiking-benchmark D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf --threshold 0.04 --tau 2.0 --steps 1000

echo.
echo --- Testing Higher Sparsity (V_th = 0.06) ---
..\build\bin\Release\popbrain_cli.exe --spiking-benchmark D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf --threshold 0.06 --tau 2.0 --steps 1000

pause
