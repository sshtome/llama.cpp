@echo off
setlocal
echo ========================================================================
echo  PopBrain Synaptic Memory - Zero-Backprop Batch Ingestion
echo ========================================================================
echo Input Dataset: wikipedia_dataset.txt
echo Output Memory:  wikipedia_memory.synapse
echo Model Dim:      2048
echo Clusters:       16
echo ========================================================================

cd /d "%~dp0"
..\build\bin\Release\popbrain_cli.exe --ingest-file wikipedia_memory.synapse wikipedia_dataset.txt --dim 2048 --clusters 16

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [+] Ingestion completed successfully!
    ..\build\bin\Release\popbrain_cli.exe --info wikipedia_memory.synapse
) else (
    echo.
    echo [ERROR] Ingestion failed with code %ERRORLEVEL%
)

pause
