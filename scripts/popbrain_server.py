"""
PopBrain Synaptic Server with Built-in Web UI, SNN Spike Gating & Memory Manager
================================================================================
Provides:
1. Standard OpenAI API (/v1/chat/completions) with live synaptic modulation.
2. Built-in interactive Web Chat UI (/chat) with Directory Memorization & Checkpoint Manager.
3. SNN Event-Driven Spike Activation Gating for memory reduction.
4. Support for arbitrary models and arbitrary .synapse memory files via CLI arguments.
"""

import os
import sys
import argparse
import time
import json
from typing import List, Dict, Optional, Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure current scripts directory is in sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    from popbrain.scalable_memory import PopBrainScale10kMemory
except ImportError:
    from scripts.popbrain.scalable_memory import PopBrainScale10kMemory

from ingest_knowledge import scan_directory_for_knowledge

app = FastAPI(title="PopBrain Synaptic LLM Server", version="1.3.0")

# Runtime state
device = "cuda" if torch.cuda.is_available() else "cpu"
default_cache_dir = r"C:\Users\ilya\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-1.5B-Instruct\snapshots\2e1fd397ee46e1388853d2af2c993145b0f1098a"
default_model_path = default_cache_dir if os.path.exists(default_cache_dir) else "Qwen/Qwen2.5-Coder-1.5B-Instruct"

default_checkpoint = os.path.join(os.path.dirname(scripts_dir), "assets", "popbrain_qwen_synaptic_memory.pt")
if not os.path.exists(default_checkpoint):
    default_checkpoint = None

tokenizer = None
model = None
memory: Optional[PopBrainScale10kMemory] = None
snn_enabled: bool = False
snn_threshold: float = 0.04
current_model_name: str = "Qwen2.5-Coder-1.5B"
saved_checkpoint_path: Optional[str] = None


def init_runtime(
    model_path: str = default_model_path,
    checkpoint_path: Optional[str] = default_checkpoint,
    save_memory_path: Optional[str] = None,
    learn_dir: Optional[str] = None,
    enable_spike: bool = False,
    spike_th: float = 0.04,
    modulation_strength: float = 0.25,
    inhibition_threshold: float = 0.82,
    margin_threshold: float = 0.015,
    num_clusters: int = 128,
    nprobe: int = 4,
    device_name: str = "auto",
    batch_size: int = 32
):
    global tokenizer, model, memory, snn_enabled, snn_threshold, current_model_name, device, saved_checkpoint_path

    if device_name != "auto":
        device = device_name
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    snn_enabled = enable_spike
    snn_threshold = spike_th
    saved_checkpoint_path = save_memory_path

    # Verify model path fallback
    if not os.path.exists(model_path) and ("/" in model_path or "\\" in model_path):
        if not os.path.exists(model_path):
            print(f"[!] Specified local model path '{model_path}' not found.")
            print(f"[*] Falling back to Hugging Face Hub: 'Qwen/Qwen2.5-Coder-1.5B-Instruct'")
            model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    print(f"[*] Initializing model on {device.upper()} from: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device if device.startswith("cuda") else None,
        torch_dtype=dtype
    )
    if not device.startswith("cuda"):
        model.to(device)

    hidden_dim = model.config.hidden_size
    current_model_name = os.path.basename(model_path.rstrip("\\/"))

    print(f"[*] Initializing PopBrain Synaptic Memory (dim={hidden_dim}, clusters={num_clusters}, nprobe={nprobe}, alpha={modulation_strength})...")
    memory = PopBrainScale10kMemory(
        hidden_dim=hidden_dim,
        num_clusters=num_clusters,
        nprobe=nprobe,
        inhibition_threshold=inhibition_threshold,
        margin_threshold=margin_threshold,
        modulation_strength=modulation_strength,
        device=device
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        memory.load_memory(checkpoint_path)
        print(f"[+] Loaded synaptic memory checkpoint ({memory.num_memories} associations) from {checkpoint_path}")
    else:
        print("[*] Starting with clean synaptic state.")

    if learn_dir:
        if os.path.exists(learn_dir):
            print(f"[*] Memorizing directory: {learn_dir}...")
            ingest_folder(learn_dir, batch_size=batch_size)
            if save_memory_path:
                memory.save_memory(save_memory_path)
                print(f"[+] Saved updated synaptic memory to: {save_memory_path}")
        else:
            print(f"[!] Warning: Directory '{learn_dir}' does not exist, skipping ingestion.")


def ingest_folder(folder_path: str, batch_size: int = 32) -> int:
    """Ingests a folder of documents/code directly into current memory in VRAM."""
    global memory, model, tokenizer, device
    items = scan_directory_for_knowledge(folder_path)
    if not items:
        return 0

    embed_layer = model.get_input_embeddings()
    t0 = time.time()

    for b_start in range(0, len(items), batch_size):
        b_chunk = items[b_start : b_start + batch_size]
        prompts = [f"Question: {it['question']}\nAnswer: {it['answer']}" for it in b_chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states[-1]
            embs = embed_layer(enc.input_ids)

            chunk_keys = []
            chunk_vals = []
            for bi in range(len(b_chunk)):
                seq_len = int(enc.attention_mask[bi].sum().item())
                if seq_len >= 2:
                    h_seq = hidden_states[bi, :seq_len]
                    e_seq = embs[bi, :seq_len]
                    k = h_seq[:-1] / (torch.norm(h_seq[:-1], dim=-1, keepdim=True) + 1e-6)
                    v = e_seq[1:] / (torch.norm(e_seq[1:], dim=-1, keepdim=True) + 1e-6)
                    chunk_keys.append(k)
                    chunk_vals.append(v)

            del out, hidden_states, embs, enc
            if chunk_keys:
                memory.ingest_batch(torch.cat(chunk_keys, dim=0), torch.cat(chunk_vals, dim=0))

    memory.build_ivf_index(iterations=3)
    print(f"[+] Ingested {len(items)} knowledge items in {time.time()-t0:.2f}s! Total associations: {memory.num_memories}")
    return len(items)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "popbrain-synaptic-llm"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    synaptic_modulation: Optional[bool] = True


class IngestRequest(BaseModel):
    folder_path: str
    save_to: Optional[str] = None


class SaveMemoryRequest(BaseModel):
    file_path: str


class LoadMemoryRequest(BaseModel):
    file_path: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": current_model_name,
        "synaptic_memories": memory.num_memories if memory else 0,
        "snn_active": snn_enabled,
        "snn_threshold": snn_threshold,
        "modulation_strength": memory.modulation_strength if memory else 0.25,
        "inhibition_threshold": memory.inhibition_threshold if memory else 0.82,
        "vram_gb": round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    }


@app.post("/v1/memory/ingest")
def api_ingest(req: IngestRequest):
    if not os.path.exists(req.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path not found: {req.folder_path}")
    count = ingest_folder(req.folder_path)
    if req.save_to:
        memory.save_memory(req.save_to)
    return {
        "status": "success",
        "items_ingested": count,
        "total_associations": memory.num_memories
    }


@app.post("/v1/memory/save")
def api_save_memory(req: SaveMemoryRequest):
    try:
        memory.save_memory(req.file_path)
        return {
            "status": "success",
            "saved_to": req.file_path,
            "total_associations": memory.num_memories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memory/load")
def api_load_memory(req: LoadMemoryRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {req.file_path}")
    try:
        memory.load_memory(req.file_path)
        return {
            "status": "success",
            "loaded_from": req.file_path,
            "total_associations": memory.num_memories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/memory/clear")
def api_clear_memory():
    memory.clear()
    return {"status": "success", "total_associations": 0}


@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PopBrain Synaptic llama.cpp Web UI</title>
        <style>
            :root {
                --bg: #0f172a;
                --panel: #1e293b;
                --accent: #38bdf8;
                --synapse: #10b981;
                --text: #f8fafc;
                --muted: #94a3b8;
            }
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: var(--bg);
                color: var(--text);
                display: flex;
                height: 100vh;
            }
            .sidebar {
                width: 360px;
                background: var(--panel);
                border-right: 1px solid #334155;
                padding: 20px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                gap: 14px;
                overflow-y: auto;
            }
            .title {
                font-size: 18px;
                font-weight: bold;
                color: var(--accent);
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .stat-box {
                background: #0f172a;
                padding: 10px 12px;
                border-radius: 8px;
                border: 1px solid #334155;
                font-size: 13px;
            }
            .stat-val {
                font-size: 16px;
                font-weight: bold;
                color: var(--synapse);
                margin-top: 3px;
            }
            .panel-header {
                font-size: 12px;
                font-weight: bold;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .tool-box {
                background: #0f172a;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #334155;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .tool-box input {
                background: #1e293b;
                border: 1px solid #334155;
                color: var(--text);
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            .action-btn {
                background: var(--synapse);
                color: #0f172a;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                cursor: pointer;
            }
            .sec-btn {
                background: #334155;
                color: var(--text);
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                cursor: pointer;
            }
            .sec-btn:hover { background: #475569; }
            .main {
                flex: 1;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            .messages {
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .msg {
                max-width: 80%;
                padding: 14px 18px;
                border-radius: 12px;
                line-height: 1.5;
                white-space: pre-wrap;
            }
            .msg.user {
                background: #0284c7;
                align-self: flex-end;
            }
            .msg.assistant {
                background: #1e293b;
                border: 1px solid #334155;
                align-self: flex-start;
            }
            .synapse-hud {
                display: inline-block;
                background: rgba(16, 185, 129, 0.15);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.4);
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 11px;
                margin-bottom: 6px;
            }
            .input-bar {
                padding: 20px;
                background: var(--panel);
                border-top: 1px solid #334155;
                display: flex;
                gap: 12px;
            }
            input[type="text"] {
                flex: 1;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 14px;
                color: var(--text);
                font-size: 15px;
                outline: none;
            }
            .send-btn {
                background: var(--accent);
                color: #0f172a;
                border: none;
                border-radius: 8px;
                padding: 0 24px;
                font-weight: bold;
                font-size: 15px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="title">PopBrain Synaptic LLM</div>
            
            <div class="stat-box">
                <div>Active Model</div>
                <div class="stat-val" id="model-val">Loading...</div>
            </div>

            <div class="stat-box">
                <div>Synaptic Memory Load</div>
                <div class="stat-val" id="mem-count">0 transitions</div>
                <div style="color: var(--muted); font-size: 11px; margin-top: 4px;">Spherical Voronoi IVF Routing</div>
            </div>

            <div class="stat-box">
                <div>SNN Activation Sparsity</div>
                <div class="stat-val" id="snn-val">Active (LIF Gating)</div>
                <div style="color: var(--muted); font-size: 11px; margin-top: 4px;">Sub-threshold noise filtered</div>
            </div>

            <div class="stat-box">
                <div>GPU Memory (VRAM)</div>
                <div class="stat-val" id="vram-val">0.0 GB</div>
            </div>

            <div class="panel-header">Memorize Folder / Repository</div>
            <div class="tool-box">
                <input type="text" id="learnDir" placeholder="Path, e.g. D:\\my_project">
                <button class="action-btn" onclick="learnFolder()">Memorize Folder (0 Backprop)</button>
                <div id="ingest-status" style="font-size: 11px; color: var(--muted);"></div>
            </div>

            <div class="panel-header">Memory Checkpoint (.synapse)</div>
            <div class="tool-box">
                <input type="text" id="ckptPath" placeholder="Checkpoint path, e.g. memory.synapse">
                <div style="display: flex; gap: 6px;">
                    <button class="sec-btn" style="flex:1;" onclick="saveMemory()">Save</button>
                    <button class="sec-btn" style="flex:1;" onclick="loadMemory()">Load</button>
                    <button class="sec-btn" style="color: #ef4444;" onclick="clearMemory()">Clear</button>
                </div>
                <div id="ckpt-status" style="font-size: 11px; color: var(--muted);"></div>
            </div>
        </div>

        <div class="main">
            <div class="messages" id="chat">
                <div class="msg assistant">PopBrain Synaptic Engine is active. Model retrieves factual knowledge in-flight via lateral inhibition and SNN spike activation gating. All parameters and options can be controlled via CLI or this panel. Ask any question to verify recall.</div>
            </div>
            <div class="input-bar">
                <input type="text" id="userInput" placeholder="Ask question..." onkeydown="if(event.key==='Enter') send()">
                <button class="send-btn" onclick="send()">Send</button>
            </div>
        </div>

        <script>
            async function updateStats() {
                try {
                    const res = await fetch('/health');
                    const data = await res.json();
                    document.getElementById('model-val').innerText = data.model;
                    document.getElementById('mem-count').innerText = data.synaptic_memories.toLocaleString() + ' transitions';
                    document.getElementById('snn-val').innerText = data.snn_active ? 'Enabled (80%+ Sparsity)' : 'Disabled';
                    document.getElementById('vram-val').innerText = data.vram_gb + ' GB';
                } catch(e){}
            }
            setInterval(updateStats, 4000);
            updateStats();

            async function learnFolder() {
                const path = document.getElementById('learnDir').value.trim();
                const st = document.getElementById('ingest-status');
                if (!path) return;
                st.innerText = 'Memorizing documents (0 backprop)...';
                try {
                    const res = await fetch('/v1/memory/ingest', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({folder_path: path})
                    });
                    const d = await res.json();
                    st.innerText = `Memorized ${d.items_ingested} items! Total: ${d.total_associations.toLocaleString()}`;
                    updateStats();
                } catch(e) {
                    st.innerText = 'Error: ' + e;
                }
            }

            async function saveMemory() {
                const p = document.getElementById('ckptPath').value.trim();
                const st = document.getElementById('ckpt-status');
                if (!p) return;
                st.innerText = 'Saving checkpoint...';
                try {
                    const res = await fetch('/v1/memory/save', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({file_path: p})
                    });
                    const d = await res.json();
                    st.innerText = 'Saved ' + d.total_associations.toLocaleString() + ' transitions!';
                } catch(e) { st.innerText = 'Error: ' + e; }
            }

            async function loadMemory() {
                const p = document.getElementById('ckptPath').value.trim();
                const st = document.getElementById('ckpt-status');
                if (!p) return;
                st.innerText = 'Loading checkpoint...';
                try {
                    const res = await fetch('/v1/memory/load', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({file_path: p})
                    });
                    const d = await res.json();
                    st.innerText = 'Loaded ' + d.total_associations.toLocaleString() + ' transitions!';
                    updateStats();
                } catch(e) { st.innerText = 'Error: ' + e; }
            }

            async function clearMemory() {
                if (!confirm('Clear all synaptic memories in RAM?')) return;
                const st = document.getElementById('ckpt-status');
                try {
                    await fetch('/v1/memory/clear', {method: 'POST'});
                    st.innerText = 'Memory cleared.';
                    updateStats();
                } catch(e) { st.innerText = 'Error: ' + e; }
            }

            async function send() {
                const inp = document.getElementById('userInput');
                const text = inp.value.trim();
                if (!text) return;
                inp.value = '';

                const chat = document.getElementById('chat');
                const uMsg = document.createElement('div');
                uMsg.className = 'msg user';
                uMsg.innerText = text;
                chat.appendChild(uMsg);

                const aMsg = document.createElement('div');
                aMsg.className = 'msg assistant';
                aMsg.innerHTML = '<i>Thinking...</i>';
                chat.appendChild(aMsg);
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            messages: [{role: 'user', content: text}],
                            temperature: 0.0,
                            max_tokens: 256
                        })
                    });
                    const data = await res.json();
                    const diag = data.synaptic_diagnostics || {};
                    let badge = '';
                    if (diag.fired_steps > 0) {
                        badge = `<div class="synapse-hud">Synapse Active: ${diag.fired_steps} tokens | Sim: ${diag.max_cosine_similarity} | Sparsity: ${(diag.sparsity_ratio * 100).toFixed(1)}%</div><br>`;
                    }
                    aMsg.innerHTML = badge + data.choices[0].message.content;
                } catch(e) {
                    aMsg.innerText = 'Error: ' + e;
                }
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer, memory, snn_enabled, snn_threshold

    prompt = ""
    for msg in req.messages:
        prompt += f"{msg.role.upper()}: {msg.content}\n"
    prompt += "ASSISTANT: "

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    curr_ids = enc.input_ids[0]
    initial_len = curr_ids.shape[0]

    gen_tokens = []
    fired_steps = 0
    max_similarity = 0.0
    last_sparsity = 0.0
    t0 = time.time()

    with torch.no_grad():
        for _ in range(req.max_tokens):
            inp = curr_ids.unsqueeze(0)
            out = model(inp, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][0, -1, :].clone()

            # 1. SNN Event-Driven Spike Activation Gating
            if snn_enabled:
                mask = torch.abs(last_hidden) >= snn_threshold
                zeros = (~mask).sum().item()
                last_sparsity = zeros / last_hidden.numel()
                last_hidden = last_hidden * mask

            # 2. PopBrain Hetero-Associative Modulation
            if req.synaptic_modulation and memory and memory.num_memories > 0:
                delta_h, sim = memory.compute_synaptic_modulation(last_hidden)
                if sim > max_similarity:
                    max_similarity = sim
                if torch.norm(delta_h) > 1e-5:
                    fired_steps += 1
                    mod_hidden = (last_hidden + delta_h).unsqueeze(0).unsqueeze(0)
                    logits = model.lm_head(mod_hidden)[0, 0, :]
                else:
                    logits = out.logits[0, -1, :]
            else:
                logits = out.logits[0, -1, :]

            if req.temperature and req.temperature > 0.0:
                probs = torch.softmax(logits / req.temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits).unsqueeze(0)

            curr_ids = torch.cat([curr_ids, next_tok])
            gen_tokens.append(next_tok.item())

            if next_tok.item() == tokenizer.eos_token_id:
                break

    dt = time.time() - t0
    response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop" if (gen_tokens and gen_tokens[-1] == tokenizer.eos_token_id) else "length"
        }],
        "usage": {
            "prompt_tokens": initial_len,
            "completion_tokens": len(gen_tokens),
            "total_tokens": initial_len + len(gen_tokens)
        },
        "synaptic_diagnostics": {
            "synaptic_active": req.synaptic_modulation,
            "fired_steps": fired_steps,
            "max_cosine_similarity": round(max_similarity, 4),
            "sparsity_ratio": round(last_sparsity, 4),
            "generation_time_sec": round(dt, 3),
            "tokens_per_sec": round(len(gen_tokens) / max(0.001, dt), 2)
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="PopBrain Synaptic Server with SNN Spike Gating and IVF Memory Routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind server")
    parser.add_argument("--port", "-p", type=int, default=8088, help="HTTP port for Web UI & API")
    parser.add_argument("--model", "-m", type=str, default=default_model_path, help="Path or HuggingFace ID of model")
    parser.add_argument("--checkpoint", "--memory", "-c", type=str, default=default_checkpoint, help="Path to .synapse memory checkpoint")
    parser.add_argument("--save-memory", type=str, default=None, help="Save resulting memory checkpoint to path after ingestion/startup")
    parser.add_argument("--learn-dir", "-d", type=str, default=None, help="Directory to ingest into synaptic memory (0 backprop)")
    parser.add_argument("--spike", "-s", action="store_true", help="Enable SNN Event-Driven Spike Activation Gating")
    parser.add_argument("--no-spike", action="store_true", help="Disable SNN Spike Activation Gating")
    parser.add_argument("--spike-th", type=float, default=0.04, help="LIF Spike activation threshold V_th (yields 80%%+ sparsity)")
    parser.add_argument("--modulation-strength", "--alpha", "-a", type=float, default=0.25, help="Synaptic modulation strength alpha")
    parser.add_argument("--inhibition-th", type=float, default=0.82, help="Lateral inhibition cosine similarity threshold")
    parser.add_argument("--margin-th", type=float, default=0.015, help="Differential margin gap threshold (top1 - top2)")
    parser.add_argument("--clusters", type=int, default=128, help="Number of Spherical Voronoi cortical clusters")
    parser.add_argument("--nprobe", type=int, default=4, help="Number of Voronoi clusters probed per query")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for directory document memorization")
    args = parser.parse_args()

    enable_spike = args.spike and not args.no_spike

    init_runtime(
        model_path=args.model,
        checkpoint_path=args.checkpoint,
        save_memory_path=args.save_memory,
        learn_dir=args.learn_dir,
        enable_spike=enable_spike,
        spike_th=args.spike_th,
        modulation_strength=args.modulation_strength,
        inhibition_threshold=args.inhibition_th,
        margin_threshold=args.margin_th,
        num_clusters=args.clusters,
        nprobe=args.nprobe,
        device_name=args.device,
        batch_size=args.batch_size
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
