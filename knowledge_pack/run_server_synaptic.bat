@echo off
setlocal
echo ========================================================================
echo  PopBrain Synaptic LLM - Native C++ Server (OpenAI Compatible)
echo ========================================================================
echo Model:    D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf
echo Memory:   wikipedia_memory.synapse
echo Endpoint: http://127.0.0.1:8080/v1
echo ========================================================================

cd /d "%~dp0"
..\build\bin\Release\llama-server.exe -m D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf --memory wikipedia_memory.synapse --port 8080 --host 127.0.0.1

pause
