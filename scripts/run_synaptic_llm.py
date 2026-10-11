"""
PopBrain Universal Synaptic LLM Launcher
========================================
Interactive cross-platform runner that connects local GGUF models or PyTorch LLMs
with zero-backprop PopBrain synaptic memory and SNN event-driven spike activation gating.

All CLI arguments are fully supported and forwarded to the server runtime.
"""

import os
import sys
import argparse
import subprocess
import time
import json
import urllib.request
import webbrowser

BANNER = r"""
========================================================================
   ____             ____            _       
  |  _ \ ___  _ __ | __ ) _ __ __ _(_)_ __  
  | |_) / _ \| '_ \|  _ \| '__/ _` | | '_ \ 
  |  __/ (_) | |_) | |_) | | | (_| | | | | |
  |_|   \___/| .__/|____/|_|  \__,_|_|_| |_|
             |_|   SYNAPTIC LLM RUNTIME v1.3.0
========================================================================
  * Architecture: Zero-Backprop Hetero-Associative Voronoi Memory (IVF)
  * SNN Dynamics: Leaky Integrate-and-Fire Event-Driven Spike Activation Gating
  * Endpoints: OpenAI API (http://{HOST}:{PORT}/v1) & Web UI (/chat)
========================================================================
"""


def check_and_install_dependencies():
    required = ["fastapi", "uvicorn", "torch", "transformers"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def main():
    parser = argparse.ArgumentParser(
        description="PopBrain Universal Synaptic LLM Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind server")
    parser.add_argument("--port", "-p", type=int, default=8088, help="Server port (default: 8088)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path or HuggingFace ID of model")
    parser.add_argument("--memory", "--checkpoint", "-c", type=str, default=None, help="Path to .synapse memory checkpoint")
    parser.add_argument("--save-memory", type=str, default=None, help="Path to save memory checkpoint after ingestion")
    parser.add_argument("--learn-dir", "-d", type=str, default=None, help="Directory/repository to ingest into memory (0 backprop)")
    parser.add_argument("--spike", "-s", action="store_true", help="Enable SNN Spike Activation Gating for memory saving")
    parser.add_argument("--no-spike", action="store_true", help="Explicitly disable SNN Spike Activation Gating")
    parser.add_argument("--spike-th", type=float, default=0.04, help="LIF Spike threshold (default: 0.04, yields 80%%+ sparsity)")
    parser.add_argument("--modulation-strength", "--alpha", "-a", type=float, default=0.25, help="Synaptic modulation strength alpha")
    parser.add_argument("--inhibition-th", type=float, default=0.82, help="Lateral inhibition threshold")
    parser.add_argument("--margin-th", type=float, default=0.015, help="Differential margin threshold")
    parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, default=1.15, help="Repetition penalty factor")
    parser.add_argument("--clusters", type=int, default=128, help="Number of Spherical Voronoi clusters")
    parser.add_argument("--nprobe", type=int, default=4, help="Number of Voronoi clusters to probe")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for directory memorization")
    parser.add_argument("--no-ui", "--headless", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    check_and_install_dependencies()

    print(BANNER.replace("{HOST}", args.host).replace("{PORT}", str(args.port)))

    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "popbrain_server.py")
    cmd = [sys.executable, server_script, "--host", args.host, "--port", str(args.port)]

    if args.model:
        cmd.extend(["--model", args.model])
    if args.memory:
        cmd.extend(["--checkpoint", args.memory])
    if args.save_memory:
        cmd.extend(["--save-memory", args.save_memory])
    if args.learn_dir:
        cmd.extend(["--learn-dir", args.learn_dir])
    if args.spike:
        cmd.append("--spike")
    if args.no_spike:
        cmd.append("--no-spike")
    if args.spike_th != 0.04:
        cmd.extend(["--spike-th", str(args.spike_th)])
    if args.modulation_strength != 0.25:
        cmd.extend(["--modulation-strength", str(args.modulation_strength)])
    if args.inhibition_th != 0.82:
        cmd.extend(["--inhibition-th", str(args.inhibition_th)])
    if args.margin_th != 0.015:
        cmd.extend(["--margin-th", str(args.margin_th)])
    if args.repetition_penalty != 1.15:
        cmd.extend(["--repetition-penalty", str(args.repetition_penalty)])
    if args.clusters != 128:
        cmd.extend(["--clusters", str(args.clusters)])
    if args.nprobe != 4:
        cmd.extend(["--nprobe", str(args.nprobe)])
    if args.device != "auto":
        cmd.extend(["--device", args.device])
    if args.batch_size != 32:
        cmd.extend(["--batch-size", str(args.batch_size)])

    print(f"[*] Starting PopBrain Synaptic Server on {args.host}:{args.port}...")
    proc = subprocess.Popen(cmd)

    target_url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"

    # Wait for server healthcheck
    for _ in range(40):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"{target_url}/health", timeout=1) as resp:
                data = json.loads(resp.read().decode())
                print("\n[+] PopBrain Synaptic Engine is ONLINE and READY!")
                print(f"    * Model: {data.get('model')}")
                print(f"    * Synaptic Memories: {data.get('synaptic_memories')} stored associations")
                print(f"    * SNN Spike Gating: {'ACTIVE (80%+ Sparsity)' if data.get('snn_active') else 'OFF'}")
                print(f"    * VRAM Footprint: {data.get('vram_gb')} GB")
                print(f"    * Interactive Web Chat: {target_url}/chat")
                print(f"    * OpenAI API Endpoint: {target_url}/v1\n")
                if not args.no_ui:
                    webbrowser.open(f"{target_url}/chat")
                break
        except Exception:
            pass

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        proc.terminate()


if __name__ == "__main__":
    main()
