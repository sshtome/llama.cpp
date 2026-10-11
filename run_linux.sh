#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  PopBrain Synaptic LLM - Zero-Backprop Local AI"
echo "========================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/scripts/run_synaptic_llm.py" "$@"
