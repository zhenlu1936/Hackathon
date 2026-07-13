#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m unittest -q c31.test_c31
python3 -m unittest -v c32.test_contract c32.test_precision_policy
python3 -m c32.test_c32
python3 -m c33.test_c33
python3 -m unittest -v c33.test_executable_fusions c33.test_fused_kernels
python3 -m c34.test_c34
python3 -m unittest -v c34.test_executable_plan
python3 -m unittest -v c35.test_c35 c35.test_cross_stage
python3 -m unittest -v c3common.test_scoring_regressions
./run_c35.sh
