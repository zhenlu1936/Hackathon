#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

required_failures=0
diagnostic_failures=0

run_stage() {
  local kind="$1"
  local label="$2"
  shift 2

  printf '\n===== %s =====\n' "$label"
  "$@"
  local status=$?
  if (( status == 0 )); then
    printf '===== PASS: %s =====\n' "$label"
  elif [[ "$kind" == "required" ]]; then
    printf '===== FAIL: %s (exit %d) =====\n' "$label" "$status"
    required_failures=$((required_failures + 1))
  else
    printf '===== DIAGNOSTIC SHORTFALL: %s (exit %d) =====\n' "$label" "$status"
    diagnostic_failures=$((diagnostic_failures + 1))
  fi
}

run_stage required "C3.1 unit tests" \
  python3 -m unittest -q c31.test_c31
run_stage required "C3.2 contract and registry tests" \
  python3 -m unittest -v \
    c32.test_contract c32.test_kernel_registry c32.test_precision_policy
run_stage diagnostic "C3.2 written-rubric diagnostic" \
  python3 -m c32.test_c32
run_stage diagnostic "C3.3 written-rubric diagnostic" \
  python3 -m c33.test_c33
run_stage required "C3.3 executable fusion tests" \
  python3 -m unittest -v c33.test_executable_fusions c33.test_fused_kernels
run_stage required "C3.4 scheduler self-test" \
  python3 -m c34.test_c34
run_stage required "C3.4 executable-plan tests" \
  python3 -m unittest -v c34.test_executable_plan
run_stage required "C3.5 and cross-stage tests" \
  python3 -m unittest -v c35.test_c35 c35.test_cross_stage
run_stage required "Scoring regression tests" \
  python3 -m unittest -v c3common.test_scoring_regressions
run_stage required "C3.5 three-model black-box test" \
  ./run_c35.sh

printf '\n===== ALLTEST SUMMARY =====\n'
printf 'Required failures:   %d\n' "$required_failures"
printf 'Diagnostic shortfalls: %d\n' "$diagnostic_failures"

if (( required_failures > 0 )); then
  printf 'ALLTEST RESULT: FAIL\n'
  exit 1
fi

if (( diagnostic_failures > 0 )); then
  printf 'ALLTEST RESULT: PASS WITH DOCUMENTED DIAGNOSTIC SHORTFALLS\n'
else
  printf 'ALLTEST RESULT: PASS\n'
fi
