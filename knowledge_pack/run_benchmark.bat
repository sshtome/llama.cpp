@echo off
setlocal
echo ========================================================================
echo  PopBrain Synaptic Memory - C++ Retrieval & Inhibition Benchmark
echo ========================================================================
cd /d "%~dp0"
..\build\bin\Release\popbrain_cli.exe --benchmark wikipedia_memory.synapse 2000
pause
