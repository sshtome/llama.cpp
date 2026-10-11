# PopBrain Synaptic & Spiking SNN Knowledge Pack: Руководство и Справочник

Настоящий пакет содержит инструменты и данные для:
1. Мгновенного дообучения любой LLM новым фактам **без бэкпропагации** (Zero-Backpropagation Synaptic Plasticity).
2. **Конвертации обычных моделей (ANN) в спайковые нейросети (SNN)** с динамикой Leaky Integrate-and-Fire (LIF) и структурной разреженностью до $89.6\%$.
3. Инспекции файлов памяти `.synapse`, проведения сравнительных тестов и запуска инференс-сервера.

---

## 1. Структура Директории

```
knowledge_pack/
├── wikipedia_dataset.txt       # Исходный текстовый датасет фактов (Формат: Ключ | Значение)
├── wikipedia_memory.synapse    # Скомпилированный бинарный файл синаптической памяти (IVF-16)
├── run_train.bat               # Батник сборки/инжестирования памяти из текстового файла
├── run_info.bat                # Батник инспекции заголовка и геометрии Вороного памяти
├── run_benchmark.bat           # Батник бенчмарка латентности выборки и латерального торможения
├── run_convert_spiking.bat     # Батник конвертации ANN модели в спайковую (SNN GGUF + LIF)
├── run_spiking_benchmark.bat   # Батник тестирования спайковой динамики, частоты импульсов и разреженности
├── run_eval_baseline.bat       # Сравнительный тест ДО подключения памяти
├── run_eval_synaptic.bat       # Сравнительный тест ПОСЛЕ подключения памяти (--memory)
├── run_server_synaptic.bat     # Запуск высокопроизводительного C++ OpenAI-сервера с памятью
└── COMMANDS.md                 # Данный справочник
```

---

## 2. Спайковая конвертация модели (ANN-to-SNN Conversion)

### Принцип работы
В классических нейросетях (ANN) активации непрерывны ($x \in \mathbb{R}$), что требует непрерывных операций умножения матриц ($W \cdot x$) при 100% плотности.
В спайковой нейросети (SNN) нейроны интегрируют мембранный потенциал $V(t)$ и генерируют дискретные импульсы (спайки) только при достижении порога $V_{\text{th}}$:
$$V(t) = V(t-1) \cdot \left(1 - \frac{1}{\tau}\right) + \frac{1}{\tau} X(t)$$
$$S(t) = \Theta(|V(t)| - V_{\text{th}}), \quad V(t) \leftarrow 0 \text{ (сброс при спайке)}$$
Подпороговые значения отсекаются, что создает **$66\% - 89.6\%$ нулей** в промежуточных слоях. Умножение матриц заменяется разреженным сложением активных столбцов, снижая энергопотребление и вычислительную сложность в $3\times - 9.6\times$ раз.

### Команда конвертации модели в SNN:
```powershell
..\build\bin\Release\popbrain_cli.exe --convert-to-spiking <input.gguf> <output.spiking.gguf> --threshold 0.04 --tau 2.0
```
*(Или запустите `run_convert_spiking.bat`)*

Конвертер:
- Считывает заголовок и тензоры исходной модели GGUF.
- Внедряет метаданные спайкового LIF-ядра (`popbrain.snn.enabled = true`, $V_{\text{th}} = 0.04$, $\tau = 2.0$).
- Генерирует согласованный файл синаптического состояния `<output>.spiking.synapse`.
- Подготавливает модель к прямому спайковому инференсу.

### Команда бенчмарка спайковой динамики:
```powershell
..\build\bin\Release\popbrain_cli.exe --spiking-benchmark <model.gguf> --threshold 0.06 --tau 2.0 --steps 1000
```
*(Или запустите `run_spiking_benchmark.bat`)*

---

## 3. Обучение без бэкпропагации (Zero-Backprop Ingestion)

### Формат данных (`wikipedia_dataset.txt`):
```text
Вопрос или контекстный ключ | Точный целевой факт или продолжение
```

### Пакетное обучение:
```powershell
..\build\bin\Release\popbrain_cli.exe --ingest-file wikipedia_memory.synapse wikipedia_dataset.txt --dim 2048 --clusters 16
```
*(Или запустите `run_train.bat`)*

### Инспекция памяти:
```powershell
..\build\bin\Release\popbrain_cli.exe --info wikipedia_memory.synapse
```
*(Или запустите `run_info.bat`)*

---

## 4. Инференс и Сервер

### Запуск CLI со спайковой моделью и памятью:
```powershell
..\build\bin\Release\llama-cli.exe -m D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf --memory wikipedia_memory.synapse -p "Ваш вопрос" --single-turn --simple-io
```

### Запуск нативного OpenAI-совместимого сервера:
```powershell
..\build\bin\Release\llama-server.exe -m D:\projects\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.spiking.gguf --memory wikipedia_memory.synapse --port 8080 --host 127.0.0.1
```
Эндпоинт: `http://127.0.0.1:8080/v1/chat/completions`
