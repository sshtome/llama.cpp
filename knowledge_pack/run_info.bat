@echo off
setlocal
echo ========================================================================
echo  PopBrain Synaptic Memory - Checkpoint Inspector
echo ========================================================================
cd /d "%~dp0"
..\build\bin\Release\popbrain_cli.exe --info wikipedia_memory.synapse
pause
