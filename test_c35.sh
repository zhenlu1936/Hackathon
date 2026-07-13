#!/usr/bin/env bash
# ============================================================
# C3.5 End-to-End Test Runner  (remote GPU server)
# ============================================================
# Development regression suite, not the written-spec black-box evaluator.
# Use ./run_c35_standard.sh for standards-oriented execution/timing/NVML.
# Run this ON the remote server (mig06), not locally.
# Uses system python3 — no venv, no pytest.
#
# Usage:
#   ./run_c35_tests.sh                             # run all tests
#   ./run_c35_tests.sh mlp                         # run only MLP tests
#   ./run_c35_tests.sh C35SpecificationTests       # run specific class
#   ./run_c35_tests.sh C35OperatorTests.test_relu  # run single test
#
# Prerequisites (one-time):
#   1. git clone                       # code + models
#   2. scp .specification/ to server   # test data
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

TEST_FILTER="${1:-}"

# ----- helpers ----------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }

# ----- hard prerequisites -----------------------------------
[ -d "models" ] || die "models/ folder not found — it should be part of the repo."
[ -d ".specification" ] || die ".specification/ folder not found. Copy it:  scp -i .ssh/mig06 -P 1106 -r .specification/testcases/release_to_competitors/testdata/ mig06@39.107.68.147:~/Hackathon/.specification/testcases/release_to_competitors/"

# ----- run tests --------------------------------------------
echo ""
echo "=== C3.5 Specification Tests ==="

if [ -n "$TEST_FILTER" ]; then
    python3 -m unittest c35.test_c35.${TEST_FILTER} -v
else
    python3 -m unittest discover -s c35 -p "test_c35.py" -v
fi
