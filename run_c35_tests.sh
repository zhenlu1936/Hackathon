#!/usr/bin/env bash
# ============================================================
# C3.5 End-to-End Test Runner  (remote GPU server)
# ============================================================
# Run this ON the remote server (mig06), not locally.
# Uses system python3 — no venv needed.
#
# Usage:
#   ./run_c35_tests.sh              # run all tests
#   ./run_c35_tests.sh -k mlp       # run only MLP tests
#   ./run_c35_tests.sh -v           # verbose output
#   ./run_c35_tests.sh --setup      # only install deps, skip tests
#
# Prerequisites (one-time):
#   1. git clone                       # code + models
#   2. scp .specification/ to server   # test data
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

# ----- helpers ----------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
ok()  { echo -e "${GREEN}OK:${NC} $*"; }

# ----- hard prerequisites -----------------------------------
[ -d "models" ] || die "models/ folder not found — it should be part of the repo."
[ -d ".specification" ] || die ".specification/ folder not found. Copy it:  scp -i .ssh/mig06 -P 1106 -r .specification/testcases/release_to_competitors/testdata/ mig06@39.107.68.147:~/Hackathon/.specification/testcases/release_to_competitors/"

# ----- install deps if needed -------------------------------
MISSING=false
python3 -c "import pytest" 2>/dev/null || MISSING=true
python3 -c "import numpy"  2>/dev/null || MISSING=true
python3 -c "import onnx"   2>/dev/null || MISSING=true
python3 -c "import torch"  2>/dev/null || MISSING=true

if $MISSING; then
    echo "Installing dependencies..."
    pip install --break-system-packages -r environment/requirements-linux-gpu.txt -q
    pip install --break-system-packages pytest numpy -q
    ok "dependencies installed"
else
    ok "dependencies already installed"
fi

# ----- done if setup-only -----------------------------------
$SETUP_ONLY && { ok "setup complete."; exit 0; }

# ----- run tests --------------------------------------------
echo ""
echo "=== C3.5 Specification Tests ==="
python3 -m pytest c35/test_c35.py -v "${PYTEST_ARGS[@]}"
