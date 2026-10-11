"""
PopBrain Universal Synaptic LLM Launcher
========================================
Interactive cross-platform runner that connects local GGUF models or PyTorch LLMs
with zero-backprop PopBrain synaptic memory.
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
  ____             ____            _       
 |  _ \ ___  _ __ | __ ) _ __ __ _(_)_ __  
 | |_) / _ \| '_ \|  _ \| '__/ _` | | '_ \ 
 |  __/ (_) | |_) | |_) | | | (_| | | | | |
 |_|   \___/| .__/|____/|_|  \__,_|_|_| |_|
            |_|   SYNAPTIC LLM RUNTIME v1.3.0
======================================================
 Infinite Associative Memory • Zero Catastrophic Forgetting
 Web UI & OpenAI-Compatible Local Inference
======================================================
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


def start_server(port: int = 8088, model_path: str = None, open_ui: bool = True):
    print(BANNER)
    print(f"[*] Starting PopBrain Synaptic Server on http://127.0.0.1:{port}...")
    
    server_script = os.path.join(os.path.dirname(__file__), "popbrain_server.py")
    if not os.path.exists(server_script):
        # Fallback to popbrain directory
        server_script = os.path.join(os.path.dirname(__file__), "popbrain", "server.py")

    cmd = [sys.executable, server_script, "--port", str(port)]
    if model_path:
        cmd.extend(["--model", model_path])

    print("[*] Initializing neural weights & synaptic cortical columns...")
    proc = subprocess.Popen(cmd)

    # Wait for healthcheck
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                data = json.loads(resp.read().decode())
                print("\n[+] PopBrain Server is ONLINE and READY!")
                print(f"    * Model: {data.get('model')}")
                print(f"    * Synaptic Memories: {data.get('synaptic_memories')} associations")
                print(f"    * VRAM Footprint: {data.get('vram_gb')} GB")
                print(f"    * OpenAI API Endpoint: http://127.0.0.1:{port}/v1")
                print(f"    * Interactive Web Chat: http://127.0.0.1:{port}/chat\n")
                if open_ui:
                    webbrowser.open(f"http://127.0.0.1:{port}/chat")
                break
        except Exception:
            pass

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="PopBrain Universal Synaptic LLM Launcher")
    parser.add_argument("--port", type=int, default=8088, help="Server port (default: 8088)")
    parser.add_argument("--model", type=str, default=None, help="Custom model path")
    parser.add_argument("--no-ui", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    check_and_install_dependencies()
    start_server(port=args.port, model_path=args.model, open_ui=not args.no_ui)


if __name__ == "__main__":
    main()
