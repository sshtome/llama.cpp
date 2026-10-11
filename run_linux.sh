#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  PopBrain Synaptic LLM - Native C++ Management CLI"
echo "========================================================"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/popbrain_cli" ]; then
    "$DIR/popbrain_cli" "$@"
elif [ -f "$DIR/build/popbrain_cli" ]; then
    "$DIR/build/popbrain_cli" "$@"
else
    echo "[ERROR] popbrain_cli not found! Please compile with CMake."
    exit 1
fi
