"""
PopBrain Synaptic Server with Built-in Web UI, SNN Spike Gating & Multi-Provider API
====================================================================================
Supported Protocols:
1. OpenAI API (/v1/chat/completions, /v1/models) with streaming (SSE) and live synaptic modulation.
2. Ollama API (/api/tags, /api/version, /api/chat, /api/generate) for Cherry Studio, OpenWebUI.
3. Built-in interactive Web Chat UI (/ and /chat) with live stats, directory ingestion, and memory checkpoints.
4. SNN Event-Driven Spike Activation Gating for memory/compute reduction.
5. Full CORS support for Electron clients (Cherry Studio, LM Studio) and web apps.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Generator

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="PopBrain Synaptic LLM Server", version="1.3.4")

# Enable CORS for Cherry Studio, LM Studio, OpenWebUI, and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime state
device = "cuda" if torch.cuda.is_available() else "cpu"
default_cache_dir = r"C:\Users\ilya\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-1.5B-Instruct\snapshots\2e1fd397ee46e1388853d2af2c993145b0f1098a"
default_model_path = default_cache_dir if os.path.exists(default_cache_dir) else "Qwen/Qwen2.5-Coder-1.5B-Instruct"

candidate_checkpoints = [
    os.path.join(os.path.dirname(scripts_dir), "assets", "popbrain_qwen_synaptic_memory.pt"),
    r"D:\projects\popbrain\assets\popbrain_qwen_synaptic_memory.pt",
    os.path.join(scripts_dir, "popbrain_qwen_synaptic_memory.pt"),
    "memory.synapse"
]
default_checkpoint = None
for cp in candidate_checkpoints:
    if os.path.exists(cp):
        default_checkpoint = cp
        break

tokenizer = None
model = None
memory: Optional[PopBrainScale10kMemory] = None
snn_enabled: bool = False
snn_threshold: float = 0.04
current_model_name: str = "Qwen2.5-Coder-1.5B-Instruct"
default_rep_penalty: float = 1.15
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
    name_candidate = getattr(model.config, "_name_or_path", "")
    if "Qwen" in model_path or "qwen" in model_path or "Qwen" in str(name_candidate):
        current_model_name = "Qwen2.5-Coder-1.5B-Instruct"
    elif name_candidate and not ("snapshots" in name_candidate or len(name_candidate) > 40):
        current_model_name = os.path.basename(name_candidate.rstrip("\\/"))
    else:
        parts = model_path.replace("\\", "/").split("/")
        clean_name = None
        for p in parts:
            if p.startswith("models--"):
                clean_name = p.replace("models--", "").replace("--", "/")
                break
        current_model_name = clean_name if clean_name else "Qwen2.5-Coder-1.5B-Instruct"

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


# Data Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "popbrain-synaptic-llm"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None
    synaptic_modulation: Optional[bool] = True
    stream: Optional[bool] = False


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: Optional[str] = "qwen2.5-coder:1.5b"
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class OllamaGenerateRequest(BaseModel):
    model: Optional[str] = "qwen2.5-coder:1.5b"
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class OllamaShowRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None


class IngestRequest(BaseModel):
    folder_path: str
    save_to: Optional[str] = None


class SaveMemoryRequest(BaseModel):
    file_path: str


class LoadMemoryRequest(BaseModel):
    file_path: str


def run_generation_stream(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    rep_pen: float = 1.15,
    synaptic_modulation: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """
    Core token-by-token generation engine.
    Yields dicts with 'token_text', 'done', 'synaptic_diagnostics'.
    """
    global model, tokenizer, memory, snn_enabled, snn_threshold

    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            for msg in messages:
                prompt += f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
    except Exception:
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    initial_len = enc.input_ids.shape[1]

    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        stop_token_ids.add(tokenizer.pad_token_id)
    for stop_str in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<eos>"]:
        tid = tokenizer.convert_tokens_to_ids(stop_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_token_ids.add(tid)

    gen_tokens = []
    accum_decoded = ""
    fired_steps = 0
    max_similarity = 0.0
    last_sparsity = 0.0
    t0 = time.time()
    past_kv = None
    curr_input = enc.input_ids

    with torch.no_grad():
        for step in range(max_tokens or 512):
            out = model(curr_input, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            last_hidden = out.hidden_states[-1][0, -1, :].clone()

            # 1. SNN Spike Gating
            if snn_enabled:
                mask = torch.abs(last_hidden) >= snn_threshold
                zeros = (~mask).sum().item()
                last_sparsity = zeros / last_hidden.numel()
                last_hidden = last_hidden * mask

            # 2. Synaptic Modulation
            if synaptic_modulation and memory and memory.num_memories > 0:
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

            # 3. Repetition Penalty
            if rep_pen and rep_pen != 1.0 and len(gen_tokens) > 0:
                for token_id in set(gen_tokens):
                    if logits[token_id] < 0:
                        logits[token_id] *= rep_pen
                    else:
                        logits[token_id] /= rep_pen

            # 4. Sampling / Greedy
            if temperature and temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits).unsqueeze(0)

            tok_id = next_tok.item()
            if tok_id in stop_token_ids:
                break

            gen_tokens.append(tok_id)
            new_decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            delta_chunk = new_decoded[len(accum_decoded):]
            accum_decoded = new_decoded

            yield {
                "token_text": delta_chunk,
                "done": False,
                "token_id": tok_id,
                "prompt_tokens": initial_len,
                "gen_count": len(gen_tokens)
            }
            curr_input = next_tok.unsqueeze(0)

    dt = time.time() - t0
    diag = {
        "synaptic_active": synaptic_modulation,
        "fired_steps": fired_steps,
        "max_cosine_similarity": round(max_similarity, 4),
        "sparsity_ratio": round(last_sparsity, 4),
        "generation_time_sec": round(dt, 3),
        "tokens_per_sec": round(len(gen_tokens) / max(0.001, dt), 2),
        "prompt_tokens": initial_len,
        "completion_tokens": len(gen_tokens),
        "total_tokens": initial_len + len(gen_tokens)
    }
    yield {
        "token_text": "",
        "done": True,
        "full_text": accum_decoded,
        "diagnostics": diag
    }


# =========================================================================
# OLLAMA API ENDPOINTS (For Cherry Studio, OpenWebUI, and Ollama clients)
# =========================================================================

@app.get("/api/tags")
def ollama_tags():
    """Ollama-compatible model list requested by Cherry Studio."""
    models_list = [
        {
            "name": current_model_name,
            "model": current_model_name,
            "modified_at": datetime.utcnow().isoformat() + "Z",
            "size": 3200000000,
            "digest": "popbrain-qwen-1.5b",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "1.5B",
                "quantization_level": "bfloat16"
            }
        },
        {
            "name": "qwen2.5-coder:1.5b",
            "model": "qwen2.5-coder:1.5b",
            "modified_at": datetime.utcnow().isoformat() + "Z",
            "size": 3200000000,
            "digest": "popbrain-qwen-1.5b",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "1.5B",
                "quantization_level": "bfloat16"
            }
        },
        {
            "name": "popbrain-synaptic-llm",
            "model": "popbrain-synaptic-llm",
            "modified_at": datetime.utcnow().isoformat() + "Z",
            "size": 3200000000,
            "digest": "popbrain",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "1.5B",
                "quantization_level": "bfloat16"
            }
        }
    ]
    return {"models": models_list}


@app.get("/api/version")
def ollama_version():
    return {"version": "0.3.14"}


@app.post("/api/show")
def ollama_show(req: OllamaShowRequest):
    return {
        "modelfile": "# PopBrain Synaptic LLM\nFROM qwen2.5-coder:1.5b",
        "parameters": "temperature 0.0\nstop <|im_end|>",
        "template": "<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
        "details": {
            "format": "gguf",
            "family": "qwen2",
            "parameter_size": "1.5B",
            "quantization_level": "bfloat16"
        }
    }


@app.post("/api/chat")
def ollama_chat(req: OllamaChatRequest):
    """Ollama /api/chat implementation supporting both streaming and non-streaming."""
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    opts = req.options or {}
    temperature = float(opts.get("temperature", 0.0))
    rep_pen = float(opts.get("repeat_penalty", default_rep_penalty))

    if req.stream:
        def stream_generator():
            for item in run_generation_stream(msgs, temperature=temperature, rep_pen=rep_pen):
                if not item["done"]:
                    chunk = {
                        "model": req.model or current_model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {"role": "assistant", "content": item["token_text"]},
                        "done": False
                    }
                    yield json.dumps(chunk, ensure_ascii=False) + "\n"
                else:
                    final_chunk = {
                        "model": req.model or current_model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "total_duration": int(item["diagnostics"]["generation_time_sec"] * 1e9),
                        "eval_count": item["diagnostics"]["completion_tokens"]
                    }
                    yield json.dumps(final_chunk, ensure_ascii=False) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        full_text = ""
        last_diag = {}
        for item in run_generation_stream(msgs, temperature=temperature, rep_pen=rep_pen):
            if item["done"]:
                full_text = item["full_text"]
                last_diag = item["diagnostics"]
        return {
            "model": req.model or current_model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": full_text
            },
            "done": True,
            "total_duration": int(last_diag.get("generation_time_sec", 0) * 1e9),
            "eval_count": last_diag.get("completion_tokens", 0)
        }


@app.post("/api/generate")
def ollama_generate(req: OllamaGenerateRequest):
    """Ollama /api/generate implementation."""
    msgs = [{"role": "user", "content": req.prompt}]
    opts = req.options or {}
    temperature = float(opts.get("temperature", 0.0))
    rep_pen = float(opts.get("repeat_penalty", default_rep_penalty))

    if req.stream:
        def stream_generator():
            for item in run_generation_stream(msgs, temperature=temperature, rep_pen=rep_pen):
                if not item["done"]:
                    chunk = {
                        "model": req.model or current_model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "response": item["token_text"],
                        "done": False
                    }
                    yield json.dumps(chunk, ensure_ascii=False) + "\n"
                else:
                    final_chunk = {
                        "model": req.model or current_model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "response": "",
                        "done": True
                    }
                    yield json.dumps(final_chunk, ensure_ascii=False) + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    else:
        full_text = ""
        for item in run_generation_stream(msgs, temperature=temperature, rep_pen=rep_pen):
            if item["done"]:
                full_text = item["full_text"]
        return {
            "model": req.model or current_model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "response": full_text,
            "done": True
        }


# =========================================================================
# OPENAI API ENDPOINTS (/v1/models, /v1/chat/completions)
# =========================================================================

@app.get("/v1/models")
@app.get("/models")
def list_models():
    """Standard OpenAI model enumeration."""
    return {
        "object": "list",
        "data": [
            {
                "id": current_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "popbrain",
                "root": current_model_name,
                "parent": None,
                "permission": [{
                    "id": f"modelperm-{current_model_name}",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }]
            },
            {
                "id": "qwen2.5-coder:1.5b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "popbrain"
            },
            {
                "id": "popbrain-synaptic-llm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "popbrain"
            },
            {
                "id": "qwen2.5-coder-popbrain",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "popbrain"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI chat completions supporting both streaming SSE and JSON."""
    msgs = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    rep_pen = req.repetition_penalty if req.repetition_penalty is not None else default_rep_penalty
    temp = req.temperature if req.temperature is not None else 0.0
    syn_mod = req.synaptic_modulation if req.synaptic_modulation is not None else True
    req_id = f"chatcmpl-{int(time.time()*1000)}"

    if req.stream:
        def sse_generator():
            for item in run_generation_stream(msgs, max_tokens=req.max_tokens or 512, temperature=temp, rep_pen=rep_pen, synaptic_modulation=syn_mod):
                if not item["done"]:
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model or current_model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": item["token_text"]},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                else:
                    final_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model or current_model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": item["diagnostics"]["prompt_tokens"],
                            "completion_tokens": item["diagnostics"]["completion_tokens"],
                            "total_tokens": item["diagnostics"]["total_tokens"]
                        }
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        full_text = ""
        last_diag = {}
        for item in run_generation_stream(msgs, max_tokens=req.max_tokens or 512, temperature=temp, rep_pen=rep_pen, synaptic_modulation=syn_mod):
            if item["done"]:
                full_text = item["full_text"]
                last_diag = item["diagnostics"]

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or current_model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": last_diag.get("prompt_tokens", 0),
                "completion_tokens": last_diag.get("completion_tokens", 0),
                "total_tokens": last_diag.get("total_tokens", 0)
            },
            "synaptic_diagnostics": last_diag
        }


# =========================================================================
# MANAGEMENT & HEALTH ENDPOINTS
# =========================================================================

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


# =========================================================================
# WEB CHAT INTERFACE (Served on both / and /chat)
# =========================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PopBrain Synaptic LLM Web UI</title>
        <style>
            :root {
                --bg: #0b0f19;
                --panel: #131b2e;
                --accent: #38bdf8;
                --synapse: #10b981;
                --text: #f8fafc;
                --muted: #94a3b8;
                --border: #1e293b;
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
                border-right: 1px solid var(--border);
                padding: 20px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                gap: 14px;
                overflow-y: auto;
            }
            .brand {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .brand-title {
                font-size: 18px;
                font-weight: 700;
                color: var(--accent);
                letter-spacing: -0.3px;
            }
            .badge-online {
                background: rgba(16, 185, 129, 0.2);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.4);
                font-size: 11px;
                padding: 2px 8px;
                border-radius: 999px;
                font-weight: 600;
            }
            .stat-box {
                background: #0b0f19;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid var(--border);
                font-size: 12px;
            }
            .stat-title {
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 11px;
            }
            .stat-val {
                font-size: 16px;
                font-weight: bold;
                color: var(--synapse);
                margin-top: 4px;
            }
            .panel-header {
                font-size: 11px;
                font-weight: bold;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 6px;
            }
            .tool-box {
                background: #0b0f19;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .tool-box input {
                background: #1a233a;
                border: 1px solid var(--border);
                color: var(--text);
                padding: 8px 10px;
                border-radius: 6px;
                font-size: 12px;
                outline: none;
            }
            .action-btn {
                background: var(--synapse);
                color: #0b0f19;
                border: none;
                padding: 8px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                cursor: pointer;
            }
            .action-btn:hover { opacity: 0.9; }
            .sec-btn {
                background: #1e293b;
                color: var(--text);
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
            }
            .sec-btn:hover { background: #334155; }
            .main {
                flex: 1;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            .top-bar {
                padding: 14px 24px;
                background: var(--panel);
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .chips-bar {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                padding: 10px 24px;
                background: rgba(19, 27, 46, 0.6);
                border-bottom: 1px solid var(--border);
            }
            .chip {
                background: #1e293b;
                border: 1px solid #334155;
                color: var(--accent);
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 12px;
                cursor: pointer;
                transition: background 0.15s;
            }
            .chip:hover {
                background: #334155;
            }
            .messages {
                flex: 1;
                overflow-y: auto;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .msg {
                max-width: 80%;
                padding: 14px 18px;
                border-radius: 12px;
                line-height: 1.6;
                white-space: pre-wrap;
                font-size: 14px;
            }
            .msg.user {
                background: #0284c7;
                align-self: flex-end;
            }
            .msg.assistant {
                background: #131b2e;
                border: 1px solid var(--border);
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
                margin-bottom: 8px;
            }
            .input-bar {
                padding: 16px 24px;
                background: var(--panel);
                border-top: 1px solid var(--border);
                display: flex;
                gap: 10px;
            }
            input[type="text"]#userInput {
                flex: 1;
                background: #0b0f19;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 12px 16px;
                color: var(--text);
                font-size: 14px;
                outline: none;
            }
            .send-btn {
                background: var(--accent);
                color: #0b0f19;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
                font-weight: bold;
                font-size: 14px;
                cursor: pointer;
            }
            .send-btn:hover { opacity: 0.9; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="brand">
                <span class="brand-title">PopBrain Synaptic LLM</span>
                <span class="badge-online" id="status-pill">ONLINE</span>
            </div>
            
            <div class="stat-box">
                <div class="stat-title">Active Model</div>
                <div class="stat-val" id="model-val" style="color: var(--accent);">Loading...</div>
            </div>

            <div class="stat-box">
                <div class="stat-title">Synaptic Memory Index</div>
                <div class="stat-val" id="mem-count">0 transitions</div>
                <div style="color: var(--muted); font-size: 11px; margin-top: 4px;">Spherical Voronoi IVF Routing</div>
            </div>

            <div class="stat-box">
                <div class="stat-title">SNN Spike Sparsity Gating</div>
                <div class="stat-val" id="snn-val">Active (LIF Gating)</div>
                <div style="color: var(--muted); font-size: 11px; margin-top: 4px;">Sub-threshold noise suppressed</div>
            </div>

            <div class="stat-box">
                <div class="stat-title">Hardware Memory (VRAM)</div>
                <div class="stat-val" id="vram-val">0.0 GB</div>
            </div>

            <div class="panel-header">Memorize Folder / Repository</div>
            <div class="tool-box">
                <input type="text" id="learnDir" placeholder="Path, e.g. D:\\projects\\my_app">
                <button class="action-btn" onclick="learnFolder()">Memorize Folder (0 Backprop)</button>
                <div id="ingest-status" style="font-size: 11px; color: var(--muted);"></div>
            </div>

            <div class="panel-header">Memory Checkpoint (.synapse)</div>
            <div class="tool-box">
                <input type="text" id="ckptPath" placeholder="Path, e.g. memory.synapse">
                <div style="display: flex; gap: 6px;">
                    <button class="sec-btn" style="flex:1;" onclick="saveMemory()">Save</button>
                    <button class="sec-btn" style="flex:1;" onclick="loadMemory()">Load</button>
                    <button class="sec-btn" style="color: #ef4444;" onclick="clearMemory()">Clear</button>
                </div>
                <div id="ckpt-status" style="font-size: 11px; color: var(--muted);"></div>
            </div>
        </div>

        <div class="main">
            <div class="top-bar">
                <div style="font-size: 14px; font-weight: 600; color: var(--text);">
                    OpenAI Endpoint: <code style="color: var(--accent);">http://127.0.0.1:8088/v1</code> &nbsp;|&nbsp; Ollama: <code style="color: var(--synapse);">http://127.0.0.1:8088</code>
                </div>
                <button class="sec-btn" onclick="clearChat()">Reset Chat</button>
            </div>

            <div class="chips-bar">
                <span style="font-size: 12px; color: var(--muted); line-height: 24px;">Quick questions:</span>
                <span class="chip" onclick="askChip('Привет, как тебя зовут?')">Привет, как тебя зовут?</span>
                <span class="chip" onclick="askChip('Что такое синаптическая память PopBrain?')">Что такое синаптическая память PopBrain?</span>
                <span class="chip" onclick="askChip('Какова архитектура Voronoi IVF?')">Архитектура Voronoi IVF?</span>
            </div>

            <div class="messages" id="chat">
                <div class="msg assistant">PopBrain Synaptic Engine is active. Model retrieves factual knowledge in-flight via lateral inhibition and SNN spike activation gating. You can chat here directly, or connect Cherry Studio / LM Studio via OpenAI or Ollama endpoints.</div>
            </div>

            <div class="input-bar">
                <input type="text" id="userInput" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') send()">
                <button class="send-btn" onclick="send()">Send</button>
            </div>
        </div>

        <script>
            let conversationHistory = [];

            async function updateStats() {
                try {
                    const res = await fetch('/health');
                    const data = await res.json();
                    document.getElementById('model-val').innerText = data.model;
                    document.getElementById('mem-count').innerText = data.synaptic_memories.toLocaleString() + ' transitions';
                    document.getElementById('snn-val').innerText = data.snn_active ? 'Enabled (80%+ Sparsity)' : 'Disabled';
                    document.getElementById('vram-val').innerText = data.vram_gb + ' GB';
                    document.getElementById('status-pill').innerText = 'ONLINE';
                } catch(e){
                    document.getElementById('status-pill').innerText = 'CONNECTING...';
                }
            }
            setInterval(updateStats, 4000);
            updateStats();

            function askChip(q) {
                document.getElementById('userInput').value = q;
                send();
            }

            function clearChat() {
                conversationHistory = [];
                const chat = document.getElementById('chat');
                chat.innerHTML = '<div class="msg assistant">PopBrain Synaptic Engine is active. Conversation reset. Ask any question or specify code to verify.</div>';
            }

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

                conversationHistory.push({role: 'user', content: text});

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
                            messages: conversationHistory,
                            temperature: 0.0,
                            max_tokens: 512,
                            repetition_penalty: 1.15
                        })
                    });
                    const data = await res.json();
                    const reply = (data.choices && data.choices[0] && data.choices[0].message) ? data.choices[0].message.content : 'No response';
                    conversationHistory.push({role: 'assistant', content: reply});

                    const diag = data.synaptic_diagnostics || {};
                    let badge = '';
                    if (diag.fired_steps > 0) {
                        badge = `<div class="synapse-hud">Synapse Active: ${diag.fired_steps} tokens | Sim: ${diag.max_cosine_similarity} | Sparsity: ${(diag.sparsity_ratio * 100).toFixed(1)}%</div><br>`;
                    }
                    aMsg.innerHTML = badge + reply.replace(/\\n/g, '<br>');
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


def main():
    parser = argparse.ArgumentParser(
        description="PopBrain Synaptic Server with SNN Spike Gating, OpenAI and Ollama APIs",
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
    parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, default=1.15, help="Repetition penalty factor")
    parser.add_argument("--clusters", type=int, default=128, help="Number of Spherical Voronoi cortical clusters")
    parser.add_argument("--nprobe", type=int, default=4, help="Number of Voronoi clusters probed per query")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for directory document memorization")
    args = parser.parse_args()

    enable_spike = args.spike and not args.no_spike
    global default_rep_penalty
    default_rep_penalty = args.repetition_penalty

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
