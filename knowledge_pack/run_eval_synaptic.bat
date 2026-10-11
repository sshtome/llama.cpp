@echo off
setlocal
echo ========================================================================
echo  TEST 2: SYNAPTIC (WITH POPBRAIN MEMORY ATTACHED)
echo ========================================================================
echo Model:  D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf
echo Memory: wikipedia_memory.synapse
echo.

cd /d "%~dp0"
echo --- Query 1: Synthetic Control Fact ---
..\build\bin\Release\llama-cli.exe -m D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf --memory wikipedia_memory.synapse -p "В каком году стартовала исследовательская миссия Астрон-42 к поясу Койпера?" --single-turn --simple-io -n 64

echo.
echo --- Query 2: Wikipedia Precision Fact ---
..\build\bin\Release\llama-cli.exe -m D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf --memory wikipedia_memory.synapse -p "Какова точная скорость света в вакууме согласно фундаментальным константам?" --single-turn --simple-io -n 64

echo.
echo --- Query 3: General Reasoning / Math ---
..\build\bin\Release\llama-cli.exe -m D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf --memory wikipedia_memory.synapse -p "Сколько будет 25 * 25?" --single-turn --simple-io -n 32

pause
