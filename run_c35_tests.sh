#!/usr/bin/env bash
# ============================================================
# C3.5 End-to-End Test Runner  (remote GPU server)
# ============================================================
# Run this ON the remote server (mig06), not locally.
#
# Usage:
#   ./run_c35_tests.sh              # auto-setup venv & run all tests
#   ./run_c35_tests.sh -k mlp       # run only MLP tests
#   ./run_c35_tests.sh -v           # verbose output
#   ./run_c35_tests.sh --setup      # only create venv & install deps
#
# Prerequisites (one-time):
#   1. Clone the repo:  git clone <repo_url>
#   2. Copy test data:  scp -r .specification/ user@mig06:~/Hackathon/
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

SETUP_ONLY=false
PYTEST_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--setup" ]; then
        SETUP_ONLY=true
    else
        PYTEST_ARGS+=("$arg")
    fi
done

# ----- hard prerequisites -----------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
ok()  { echo -e "${GREEN}OK:${NC} $*"; }
warn(){ echo -e "${YELLOW}WARNING:${NC} $*"; }

if [ ! -d "models" ]; then
    die "models/ folder not found — it should be part of the repo."
fi

if [ ! -d ".specification" ]; then
    die ".specification/ folder not found. Copy it:  scp -r .specification/ mig06@39.107.68.147:~/Hackathon/"
fi

# ----- venv setup -------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
fi

# ----- install dependencies ---------------------------------
NEEDS_PYTEST=false
NEEDS_DEPS=false
python -c "import pytest" 2>/dev/null || NEEDS_PYTEST=true
python -c "import numpy"  2>/dev/null || NEEDS_DEPS=true
python -c "import onnx"   2>/dev/null || NEEDS_DEPS=true
python -c "import torch"  2>/dev/null || NEEDS_DEPS=true

if $NEEDS_DEPS || $NEEDS_PYTEST; then
    echo "Installing dependencies..."
    pip install -r environment/requirements-linux-gpu.txt -q
    pip install pytest numpy -q
    ok "dependencies installed"
else
    ok "dependencies already installed"
fi

# ----- done if setup-only -----------------------------------
if $SETUP_ONLY; then
    ok "venv ready, deps installed."
    exit 0
fi

# ----- run tests --------------------------------------------
echo ""
echo "=== C3.5 Specification Tests ==="
python -m pytest c35/test_c35.py -v "${PYTEST_ARGS[@]}"
