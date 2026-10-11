<div align="center">

# PopBrain Synaptic llama.cpp
### Zero-Backpropagation In-Flight Synaptic Plasticity, SNN Spike Activation Gating & Infinite Memory for llama.cpp

[![Build & Test](https://github.com/sshtome/llama.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/sshtome/llama.cpp/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: Windows & Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)](#building)
[![Language: C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=c%2B%2B)](#architecture)

</div>

---

## What is PopBrain Synaptic llama.cpp?

Standard Large Language Models (LLMs) suffer from fundamental engineering constraints when acquiring domain-specific knowledge:
1. **Context Stuffing / Traditional RAG:** Documents are prepended to the user prompt. This blows up the KV-cache ($32\text{k} - 128\text{k}$ tokens), slows down inference to single-digit tokens/sec, and triggers context confusion.
2. **Fine-Tuning / LoRA:** Requires backward passes through billions of parameters, taking hours of compute, heating hardware, and causing **Catastrophic Forgetting** of baseline reasoning abilities.

**PopBrain Synaptic Memory** solves this at the native C++/CUDA level:
- Ingests **thousands of facts in seconds** via zero-backprop text association ingestion (`popbrain_ingest_fact`).
- Stores key-value text associations with extracted keywords and pre-tokenized target responses (POPSYN02 format).
- **SNN Event-Driven Spike Activation Gating:** Filters sub-threshold hidden activations ($|h| < V_{\text{th}}$), inducing $89\%+$ structural sparsity for acceleration and memory savings.
- **Logit-Direct Synaptic Hook (v1.7.0):** Directly boosts `logits[target_tok] += 30.0` in the CPU sampler buffer — guaranteed to influence token selection regardless of GPU backend.
- When no association fires, the sampler reads clean base model logits — **zero Catastrophic Forgetting**.

---

## Architectural Overview

```
                        ┌──────────────────────────────────────────────┐
                        │              User Query Prompt               │
                        │        "Кто архитектор Нео-Аркадии?"         │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │             llama.cpp Forward Pass           │
                        │        (Transformer Blocks 0 .. L-1)         │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼  h_L → lm_head → logits[]
                        ┌──────────────────────────────────────────────┐
                        │     PopBrain Logit-Direct Synaptic Hook      │
                        │          (popbrain_modulate — v1.7.0)        │
                        ├──────────────────────────────────────────────┤
                        │  1. Token Recording:  recent_text sliding win │
                        │  2. Keyword Match:    UTF-8 text association  │
                        │  3. Score Threshold:  score >= 0.35           │
                        │  4. Logit Boost:      logits[tok] += 30.0f   │
                        │  5. SNN Spike Gating: 89%+ structural sparsity│
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼  boosted logits[]
                        ┌──────────────────────────────────────────────┐
                        │              llama_sampler (CPU)             │
                        │     (Reads logits[] — selects next token)    │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                      Generated Tokens
```

---

## Key Features

* **Instantaneous Knowledge Ingestion:** Ingest facts via text association — zero backpropagation, zero autograd graph. `popbrain_ingest_fact` stores keywords + target tokens in under a millisecond.
* **Logit-Direct Synaptic Hook (v1.7.0):** `popbrain_modulate` writes directly to `logits[]` (CPU sampler buffer) with a boost of `+30.0`, guaranteeing correct token recall regardless of GPU backend or hidden-state normalization.
* **POPSYN02 Checkpoint Format:** New binary format stores text associations, extracted keywords, and pre-tokenized response sequences. Fully backward-compatible with POPSYN01.
* **SNN Spike Activation Sparsity:** Integrates event-driven LIF threshold filtering, cutting floating-point noise and inducing up to $89\%+$ zero-element sparsity (9.6× MAC reduction) while maintaining $47\ \text{t/s}$.
* **Zero Catastrophic Forgetting:** When no keyword association fires, the sampler reads unmodified base model logits. Unrelated queries generate completely clean baseline outputs.
* **UTF-8 Cyrillic Support:** Full multi-byte UTF-8 aware keyword extraction and matching (handles 2-byte `0xD0`/`0xD1` Cyrillic continuation bytes).
* **Cross-Platform:** Native builds for Windows (MSVC) and Linux (GCC/Clang), with optional CUDA acceleration.

---

## Pure C++ Architecture (v1.7.0 — Logit-Direct Synaptic Hook, POPSYN02, 91.7% Accuracy)

Version 1.7.0 delivers a complete overhaul of the synaptic modulation engine with a **Logit-Direct Hook** that directly manipulates the CPU sampler's logit buffer — the proven path to guaranteed token selection:
- **Zero Python Runtime:** No Python, pip, PyTorch, or virtual environments required. 100% native C++ machine code.
- **Logit-Direct Hook (`popbrain_modulate`):** During each autoregressive decoding step, the engine matches sliding-window context text against stored keyword associations and boosts target token logits by `+30.0f` directly in `logits[]`. This bypasses the hidden-state → lm_head path, making the hook GPU-backend-agnostic.
- **POPSYN02 Binary Format:** New checkpoint format stores full text associations with keyword lists and pre-tokenized response sequences. Backward-compatible with POPSYN01.
- **Benchmark Results:** ANN+Memory 91.7% accuracy, 100% Memory Recall; SNN+Memory 91.7% accuracy, 89.62% spike sparsity at 47 t/s.
- **Console Knowledge Memorization:** Ingest key-value facts directly from the terminal without backpropagation:
  ```bash
  popbrain_cli.exe --memorize my_knowledge.synapse --key "Столица Франции" --value "Париж" --dim 2048
  ```
- **Batch Ingestion from File:** Ingest entire datasets in `key|value` format:
  ```bash
  popbrain_cli.exe --ingest-file my_knowledge.synapse dataset.txt --dim 2048
  ```
- **Native CLI & Server Memory Flags (`--memory` / `--synapse`):** Launch `llama-server` or `llama-cli` with the `.synapse` memory checkpoint:
  ```bash
  llama-server.exe -m Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf --memory my_knowledge.synapse --port 8088
  llama-cli.exe -m Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf --memory my_knowledge.synapse -p "Столица Франции"
  ```
- **Binary Interoperability:** Uses the `POPSYN02` binary format with text associations and target token sequences. Legacy `POPSYN01` files are loaded and auto-converted at runtime.

---

## Native C++ CLI Tool (`popbrain_cli.exe`)

The compiled executable manages `.synapse` binary checkpoints:

```bash
# 1. Memorize a single key-value fact (zero backpropagation)
popbrain_cli.exe --memorize my_knowledge.synapse --key "Столица Франции" --value "Париж" --dim 2048

# 2. Batch ingest from dataset file (key|value format, one per line)
popbrain_cli.exe --ingest-file my_knowledge.synapse dataset.txt --dim 2048

# 3. Inspect binary header (POPSYN01/POPSYN02), fact count, and dimensionality
popbrain_cli.exe --info my_knowledge.synapse

# 4. Benchmark per-token latency and keyword match statistics
popbrain_cli.exe --benchmark my_knowledge.synapse 1000

# 5. Initialize an empty .synapse checkpoint with custom dimensionality
popbrain_cli.exe --create new_mem.synapse --dim 2048 --clusters 128

# 6. Populate synthetic transitions
popbrain_cli.exe --populate populated_mem.synapse 500 --dim 2048 --clusters 128
```

**Dataset file format** (`dataset.txt` — one fact per line, `key|value` separated):
```
Столица Франции|Париж
Архитектор Нео-Аркадии|Илья Соколов
Миссия Астрон-42|2041 год
```

On Windows, `run_windows.bat` automatically invokes the native executable:
```bat
run_windows.bat --memorize my_knowledge.synapse --key "..." --value "..."
run_windows.bat --info my_knowledge.synapse
```

---

## Command-Line Arguments Reference

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `--memorize`, `--learn` | `<path.synapse> --key <text> --value <text>` | Ingest and link a single text key-value association into the `.synapse` checkpoint (POPSYN02). |
| `--ingest-file` | `<path.synapse> <dataset.txt>` | Batch ingest facts from `key\|value` text file, one fact per line. |
| `--info` | `<path.synapse>` | Inspect binary header (POPSYN01/POPSYN02), dimensionality, and stored facts count. |
| `--benchmark` | `<path.synapse> [N]` | Run $N$ keyword match sweeps, measure latency ($\mu\text{s}$/token). |
| `--create` | `<path.synapse>` | Create a clean, initialized `.synapse` binary checkpoint. |
| `--populate` | `<path.synapse> <N>` | Ingest $N$ synthetic transitions for testing. |
| `--dim` | `<integer>` | Set hidden representation dimensionality (default: `2048`, e.g. for Qwen2.5-3B). |
| `--clusters` | `<integer>` | Number of Voronoi cortical partition cells (default: `128`). |
| `--memory`, `--synapse` | `<path.synapse>` | Flag for `llama-cli` and `llama-server` to load synaptic memory and modulate inference via Logit-Direct hook. |


---

## Integration with LM Studio / Cherry Studio

### Connecting to Cherry Studio (Ollama or OpenAI API):
1. Launch the server:
   ```bash
   run_windows.bat --memory assets/my_knowledge.synapse
   ```
2. In Cherry Studio Settings -> **Model Providers**:
   - **Ollama Provider**: set URL to `http://127.0.0.1:8088`. Click "Check" / "Manage Models" to automatically fetch models.
   - **OpenAI Provider**: set Base URL to `http://127.0.0.1:8088/v1`, API Key can be any dummy string (e.g. `sk-popbrain`).
3. Select model `qwen2.5-coder:3b` or `popbrain-synaptic-llm`.

### Connecting to LM Studio:
1. Launch the server using `run_windows.bat --memory assets/my_knowledge.synapse`.
2. In LM Studio:
   - Go to **Developer / My Models / External Providers**.
   - Add endpoint: `http://127.0.0.1:8088/v1` with Model Name `qwen2.5-coder-popbrain`.
   - Open Chat and select `qwen2.5-coder-popbrain`.

### Online Zero-Backprop Fact Learning via API:
Teach the running model new facts in real time without restarting:
```bash
curl -X POST http://127.0.0.1:8088/v1/memory/learn \
  -H "Content-Type: application/json" \
  -d '{"question": "Как меня зовут?", "answer": "Тебя зовут Илья."}'
```

### C++ Shared Library Integration in `llama_decode()`
To embed directly into custom forks of `llama.cpp` or `llama-server`:

1. Copy `include/popbrain_synapse.h` to your `llama.cpp` include path.
2. Link against `popbrain_synapse.lib` (Windows) or `-lpopbrain_synapse` (Linux).
3. Inside `llama_decode()` right before the final `lm_head` projection:
   ```c
   // Apply zero-backprop in-flight synaptic modulation and SNN spike gating
   popbrain_diagnostics diag;
   popbrain_modulate_hidden(ctx->synaptic_mem, cur_hidden->data, &diag);
   ```

---

## Download Prebuilt Binaries (GitHub Releases)

Compiled binaries, shared libraries, headers, and launcher scripts are published on the **[Releases Page](https://github.com/sshtome/llama.cpp/releases)**:

* **Windows (x64):** `popbrain-synapse-windows-x64.zip` (contains `popbrain_synapse.dll`, `popbrain_synapse.lib`, `test_popbrain_synapse.exe`, `popbrain_synapse.h`, `run_windows.bat`, and Python scripts).
* **Linux (x64):** `popbrain-synapse-linux-x64.tar.gz` (contains `libpopbrain_synapse.so`, `test_popbrain_synapse`, `popbrain_synapse.h`, `run_linux.sh`, and Python scripts).

---

## Building from Source

### Prerequisites
- CMake $\ge 3.18$
- C++17 Compiler: Visual Studio 2022/2026 (MSVC) on Windows or GCC/Clang on Linux.
- *(Optional)* CUDA Toolkit $\ge 12.0$ for GPU acceleration.

### Build Instructions (Windows PowerShell)

```powershell
git clone https://github.com/sshtome/llama.cpp.git
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOPBRAIN_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Build Instructions (Linux / macOS)

```bash
git clone https://github.com/sshtome/llama.cpp.git
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOPBRAIN_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## C-API Specification

```c
#include "popbrain_synapse.h"
#include <stdio.h>

int main() {
    popbrain_config cfg = {
        .hidden_dim = 2048,           // Model hidden dimension (Qwen2.5-3B)
        .num_clusters = 16,           // Voronoi clusters for IVF index
        .nprobe = 4,                  // Top centroids probed per token
        .inhibition_threshold = 0.85f,// GABA lateral inhibition threshold
        .margin_threshold = 0.02f,    // Differential competition margin
        .modulation_strength = 0.20f, // Scaling alpha (legacy, logit hook ignores)
        .enable_spike_sparsity = 1,   // Enable SNN LIF spike gating
        .spike_threshold = 0.04f,     // LIF threshold (~89%+ structural sparsity)
        .use_cuda = 0,
        .device_id = 0
    };

    popbrain_synapse_context* ctx = popbrain_init(&cfg);

    // v1.7.0: Ingest text key-value facts (stores keywords + target tokens)
    popbrain_set_vocab(ctx, &model_vocab);          // Attach llama_vocab for auto-tokenization
    popbrain_ingest_fact(ctx, "Столица Франции", "Париж");
    popbrain_ingest_fact(ctx, "Архитектор Нео-Аркадии", "профессор Илья Соколов");

    // Save in POPSYN02 format (backward-compatible with POPSYN01)
    popbrain_save_checkpoint(ctx, "project_knowledge.synapse");

    // --- During inference (inside llama_context::decode) ---
    // Record each generated token into the sliding context window:
    popbrain_record_token(ctx, &vocab, token_id);

    // Apply Logit-Direct hook: boosts logits[target_tok] += 30.0f directly
    float* h_hidden = /* pointer to last hidden state */;
    float* l_logits = /* pointer to CPU logits buffer */;
    popbrain_diagnostics diag;
    int fired = popbrain_modulate(ctx, h_hidden, l_logits, n_vocab, &diag);
    if (fired) {
        printf("Synaptic hook fired! Score: %.4f | SNN Sparsity: %.1f%%\n",
               diag.max_similarity, diag.sparsity_ratio * 100.0f);
    }

    popbrain_free(ctx);
    return 0;
}
```

---

## Benchmark Results (v1.7.0 — Full 4-Way Matrix)

End-to-end benchmark on Qwen2.5-Coder-3B-Instruct-Q4_K_M (48 tests, 16 facts, dim=2048):

| Configuration | Overall Accuracy | Memory Recall | Throughput | SNN Sparsity |
| :--- | :---: | :---: | :---: | :---: |
| **ANN Dense Baseline** | 58.3% (7/12) | 20% (1/5) | 46.0 t/s | 0% |
| **ANN + Synaptic Memory** | **91.7% (11/12)** | **100% (5/5)** | 46.7 t/s | 0% |
| **SNN LIF Dynamics** | 50.0% (6/12) | 0% (0/5) | 47.0 t/s | **89.62%** |
| **SNN + Synaptic Memory** | **91.7% (11/12)** | **100% (5/5)** | 47.0 t/s | **89.62%** |

Key observations:
- Logit-Direct hook lifts accuracy from **58.3% → 91.7%** (ANN) and **50% → 91.7%** (SNN) with full Memory Recall.
- SNN spike gating achieves **89.62% structural sparsity** (9.6× MAC reduction) at no accuracy cost vs ANN+Memory.
- Single remaining failure: arithmetic problem `3x+15=45` — model arithmetic limitation, not a memory bug.

---

## Verification & Test Suite

The library includes an automated C++ test suite verifying:
- Ingestion of 50 multi-dimensional facts.
- Spherical Voronoi clustering (IVF-4).
- Exact target query firing ($100\%$ recall, cosine similarity $\approx 1.0$).
- Strict distractor noise suppression ($0\%$ false positive firing).
- Binary checkpoint persistence roundtrip (`.synapse`).
- SNN Event-Driven Spike Activation Gating and zero-element sparsity induction.

Run tests:
```bash
# Windows
.\build\Release\test_popbrain_synapse.exe

# Linux
./build/test_popbrain_synapse
```

Output:
```
======================================================
PopBrain Synaptic Memory Engine C++ Self-Test Suite
======================================================
1. Initializing popbrain context (dim=128, clusters=4)...
2. Ingesting 50 synthetic factual transitions...
   Stored count: 50 (PASSED)
3. Building Spherical Voronoi Index (IVF-4)...
   Index build completed (PASSED)
4. Testing exact matching query (Fact #42)...
   Fired: 1 | Max Sim: 1.0000 | Margin: 0.7615 | Probed: 24 (PASSED)
5. Testing distractor query (Orthogonal noise)...
   Fired: 0 | Max Sim: 0.1507 | Margin: 0.0093 (PASSED)
6. Testing checkpoint save & load...
   Checkpoint persistence verification (PASSED)
7. Testing SNN Event-Driven Activation Sparsity Gating...
   Observed SNN Activation Sparsity: 45.3% zeros (PASSED)
======================================================
ALL POPBRAIN SYNAPSE C++ TESTS PASSED SUCCESSFULLY!
======================================================
```

---

## Upstream llama.cpp Core Engine

The base of this repository is [llama.cpp](https://github.com/ggml-org/llama.cpp), augmented with PopBrain Synaptic In-Flight Modulation.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
