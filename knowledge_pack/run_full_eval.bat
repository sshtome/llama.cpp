@echo off
setlocal
echo ========================================================================
echo  PopBrain 4-Way Comprehensive Matrix Benchmark
echo  (Ordinary vs Trained vs Spiking vs Spiking+Memory)
echo ========================================================================

cd /d "%~dp0"
python eval_matrix.py

pause
