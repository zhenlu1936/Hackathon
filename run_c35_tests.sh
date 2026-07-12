#!/usr/bin/env bash
# ============================================================
# C3.5 End-to-End Test Runner
# ============================================================
# Usage:
#   ./run_c35_tests.sh              # run all tests
#   ./run_c35_tests.sh -k mlp       # run only MLP tests
#   ./run_c35_tests.sh -v           # verbose output
#
# Prerequisites:
#   - specification/ folder present (gitignored, copy separately)
#   - Python 3.12 with deps from environment/requirements-linux-gpu.txt
#   - NVIDIA GPU with driver >= 580.126.20 (for AEC GPU execution)
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ----- check prerequisites ---------------------------------
if [ ! -d "specification" ]; then
    echo "ERROR: specification/ folder not found."
    echo "  It is gitignored — copy it from the competition materials:"
    echo "  scp -r specification/ user@server:/path/to/Hackathon/"
    exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d ".venv" ]; then
        echo "Activating .venv ..."
        source .venv/bin/activate
    else
        echo "WARNING: no virtual environment active and no .venv/ found."
        echo "  python3 -m venv .venv && source .venv/bin/activate"
        echo "  pip install -r environment/requirements-linux-gpu.txt"
    fi
fi

# ----- run tests -------------------------------------------
echo "=== C3.5 Specification Tests ==="
python -m pytest c35/test_c35.py -v "$@"
