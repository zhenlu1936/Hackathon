#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Written-spec C3.5 black-box runner. Override COMMAND_TEMPLATE with the exact
# registered submission command when testing another implementation.
PYTHON="${PYTHON:-python3}"
if [[ -z "${COMMAND_TEMPLATE:-}" ]]; then
  COMMAND_TEMPLATE="$PYTHON -m c35.deploy --onnx {onnx} --input {input} --output {output} --batch-size {batch_size}"
fi

RELEASE=".specification/testcases/release_to_competitors"
[[ -d "$RELEASE/models" ]] || { echo "missing $RELEASE/models" >&2; exit 2; }
[[ -d "$RELEASE/testdata/c35" ]] || { echo "missing $RELEASE/testdata/c35" >&2; exit 2; }
REPORT="${C35_REPORT:-c35-standard-report.json}"

exec "$PYTHON" -m c35.standard_runner \
  --command "$COMMAND_TEMPLATE" \
  --report "$REPORT" \
  "$@"
