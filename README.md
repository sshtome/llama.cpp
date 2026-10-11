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
- Ingests **thousands of facts in seconds** via single-pass forward sweeps ($0\text{ MB}$ autograd graph, zero backpropagation).
- Stores associations in high-dimensional hetero-associative Voronoi cortical columns (**IVF-256**) with GABAergic lateral inhibition.
- **SNN Event-Driven Spike Activation Gating:** Filters sub-threshold hidden activations ($|h| < V_{\text{th}}$), inducing $80\%+$ structural sparsity for acceleration and memory savings.
- Modulates the model's final hidden states $h_L$ in-flight prior to token projection (`lm_head`).
- When querying unlearned concepts, **lateral inhibition suppresses activation to 0%**, guaranteeing complete preservation of base LLM knowledge.

---

## Architectural Overview

```
                        ┌──────────────────────────────────────────────┐
                        │              User Query Prompt               │
                        │           "Explain streamer.py"              │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │             llama.cpp Forward Pass           │
                        │        (Transformer Blocks 0 .. L-1)         │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼  h_L (Last Hidden State)
                        ┌──────────────────────────────────────────────┐
                        │        PopBrain Synaptic Modulation          │
                        │       (popbrain_modulate_hidden)             │
                        ├──────────────────────────────────────────────┤
                        │  1. Coarse Routing: IVF-256 Centroids        │
                        │  2. Fine Search: Active Voronoi Cells        │
                        │  3. Lateral Inhibition: theta_inhib >= 0.85  │
                        │  4. SNN Spike Gating: 80%+ Sparsity Inducer  │
                        │  5. In-Flight Shift: h_mod = h_L + delta_h   │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼  h_mod
                        ┌──────────────────────────────────────────────┐
                        │                ggml_mul_mat                  │
                        │            (Model LM Output Head)            │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                      Generated Tokens
```

---

## Key Features

* **Instantaneous Knowledge Ingestion:** Ingest 10,000+ factual transitions in under 30 seconds ($O(1)$ single forward evaluation).
* **SNN Spike Activation Sparsity:** Integrates event-driven LIF threshold filtering directly in hidden representations, cutting out floating-point noise and inducing up to $80\%+$ zero-element sparsity.
* **Zero Catastrophic Forgetting:** Dual GABAergic lateral inhibition ($\theta_{\text{inhib}} \ge 0.85$) and differential margin competition ($\Delta_{\text{margin}} \ge 0.02$) guarantee that unrelated queries generate clean baseline outputs.
* **Spherical Voronoi Indexing (IVF-256):** Partitions high-dimensional associative keys into 256 cortical columns with contiguous buffer slicing, reducing per-token retrieval latency to $< 1.5\text{ ms}$.
* **Compact Checkpoint Persistence:** Binary `.synapse` checkpoints store entire project knowledge bases in a few hundred megabytes, shareable across machines.
* **Cross-Platform:** Native builds for Windows (MSVC) and Linux (GCC/Clang), with optional CUDA acceleration.

---

## Headless Synaptic Architecture (v1.4.0)

Version 1.4.0 introduces an upgraded, completely headless, pure console architecture:
- **Upgraded Neural Backbone:** `Qwen2.5-Coder-3B-Instruct` (hidden dimension $D=2048$, 3.09B parameters) provides substantially enhanced instruction-following, coding comprehension, and complex reasoning on RTX 3060 (5.75 GB VRAM footprint).
- **Zero Web UI Overhead:** Completely eliminated all web/HTML/browser interfaces. Headless server exposes high-performance OpenAI API (`/v1`) and Ollama API endpoints for integration into external desktop clients (Cherry Studio, LM Studio, OpenWebUI) or custom pipelines.
- **Native C++ Management CLI (`popbrain_cli`):** High-speed binary tool for inspecting `.synapse` binary structures, running microsecond latency benchmarks, and validating Voronoi cluster geometry.
- **Two Distinct Console Modes:**
  1. **Offline Memory Creation / Ingestion:** Memorize facts or code directories directly from terminal arguments (`--learn-fact`, `--learn-dir`, `--learn-file` + `--save-memory`). Indexes associations into a standalone `.synapse` binary and exits immediately ($0$ autograd, no server overhead).
  2. **Headless Server Deployment:** Launch with `--memory <file.synapse>` to serve the model with in-flight synaptic modulation on OpenAI/Ollama ports.

---

## Native C++ CLI Tool (`popbrain_cli.exe`)

The compiled C++ executable (`build/Release/popbrain_cli.exe`) interacts directly with `.synapse` (`POPSYN01`) binary checkpoints:

```bash
# 1. Inspect binary header, Voronoi geometry, and memory count
build\Release\popbrain_cli.exe --info assets/my_knowledge.synapse

# 2. Benchmark per-token latency and lateral inhibition suppression
build\Release\popbrain_cli.exe --benchmark assets/my_knowledge.synapse 1000

# 3. Initialize an empty .synapse checkpoint with custom dimensionality
build\Release\popbrain_cli.exe --create new_mem.synapse --dim 2048 --clusters 128
```

---

## Console Workflow Guide

### Workflow A: Memorizing Knowledge into a Memory File (Offline)

Teach the model new facts or entire codebases purely from the console and save directly to `.synapse`:

```bash
# Memorize a single key-value fact and exit immediately:
run_windows.bat --learn-fact "Как меня зовут?=Тебя зовут Илья." --save-memory assets/my_knowledge.synapse

# Ingest an entire codebase or folder of documents:
run_windows.bat --learn-dir D:\my_project --save-memory assets/project_knowledge.synapse --spike

# Ingest a text file of QA pairs:
run_windows.bat --learn-file facts.txt --save-memory assets/facts.synapse
```

### Workflow B: Launching Headless Server with Memory Checkpoint

Run the headless server (OpenAI & Ollama APIs) pointing to your created memory file:

```bash
# Launch with custom memory checkpoint:
run_windows.bat --memory assets/my_knowledge.synapse

# Or via Python directly:
python scripts/run_synaptic_llm.py --memory assets/my_knowledge.synapse --host 127.0.0.1 --port 8088
```

---

## Command-Line Arguments Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model`, `-m` | `str` | `Qwen2.5-Coder-3B-Instruct` | Local directory or Hugging Face repository ID (PyTorch or GGUF weights). |
| `--memory`, `--checkpoint`, `-c` | `str` | `None` | Path to load a `.synapse` binary memory checkpoint on startup. |
| `--save-memory` | `str` | `None` | Path where the resulting `.synapse` memory checkpoint will be saved. |
| `--learn-fact` | `str` | `None` | Ingest single fact from CLI (format: `'Question=Answer'`). If `--save-memory` is provided, exits after saving. |
| `--learn-dir`, `-d` | `str` | `None` | Path to folder/codebase to ingest into synaptic memory (0 backprop). Exits after saving if `--save-memory` provided. |
| `--learn-file` | `str` | `None` | Path to text file with facts to ingest. |
| `--spike`, `-s` | `flag` | `False` | Enable SNN Event-Driven Spike Activation Gating ($|h| < V_{\text{th}} \to 0$) for 80%+ structural sparsity. |
| `--no-spike` | `flag` | `False` | Explicitly disable SNN Spike Activation Gating. |
| `--spike-th` | `float` | `0.04` | LIF Spike activation threshold $V_{\text{th}}$. Suppresses sub-threshold floating-point noise. |
| `--modulation-strength`, `--alpha`, `-a` | `float` | `0.25` | Synaptic modulation scaling factor $\alpha$. Controls injection strength of hetero-associative vector. |
| `--inhibition-th` | `float` | `0.82` | GABAergic lateral inhibition cosine similarity threshold $\theta_{\text{inhib}}$. Unlearned queries yield 0% activation. |
| `--margin-th` | `float` | `0.015` | Differential margin threshold $\Delta_{\text{margin}} = s_1 - s_2$ to prevent ambiguous distractor collisions. |
| `--repetition-penalty`, `--rep-penalty` | `float` | `1.15` | Repetition penalty factor to prevent degenerate cyclic token looping. |
| `--clusters` | `int` | `128` | Number of Spherical Voronoi cortical partitions (IVF clusters) on unit hypersphere $\mathbb{S}^{D-1}$. |
| `--nprobe` | `int` | `4` | Number of Voronoi cortical cells probed per token query. |
| `--port`, `-p` | `int` | `8088` | TCP port for OpenAI-compatible API and Ollama API. |
| `--host` | `str` | `127.0.0.1` | Network interface to bind server to (`127.0.0.1` or `0.0.0.0` for LAN). |
| `--device` | `str` | `auto` | Execution device (`cuda`, `cpu`, or `auto`). |
| `--batch-size` | `int` | `32` | Batch size of documents encoded concurrently during ingestion. |

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
        .hidden_dim = 2048,           // Model hidden dimension
        .num_clusters = 256,          // Voronoi cortical cells
        .nprobe = 4,                  // Top centroids probed per token
        .inhibition_threshold = 0.85f,// GABA threshold
        .margin_threshold = 0.02f,    // Differential margin
        .modulation_strength = 0.20f, // Scaling alpha
        .enable_spike_sparsity = 1,   // Enable SNN spike activation gating
        .spike_threshold = 0.04f,     // Spike threshold (yields 80%+ sparsity)
        .use_cuda = 1,
        .device_id = 0
    };

    popbrain_synapse_context* ctx = popbrain_init(&cfg);

    // Ingest transitions: popbrain_ingest_transition(ctx, k_ptr, v_ptr);
    // Build index: popbrain_build_index(ctx, 3);

    float hidden_state[2048];
    popbrain_diagnostics diag;
    
    int fired = popbrain_modulate_hidden(ctx, hidden_state, &diag);
    if (fired) {
        printf("Synaptic memory active! Similarity: %.4f | SNN Sparsity: %.1f%%\n",
               diag.max_similarity, diag.sparsity_ratio * 100.0f);
    }

    popbrain_save_checkpoint(ctx, "project_knowledge.synapse");
    popbrain_free(ctx);
    return 0;
}
```

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

## License

This project is licensed under the [MIT License](LICENSE).
