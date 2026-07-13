#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ── C3.5 ResNet Deep Profiler ──────────────────────────────────────
# Instruments every stage of the C3.1→C3.5 pipeline for the ResNet model
# and writes a detailed profile report.
#
# Usage:
#   ./run_resnet_profile.sh                  # default batch 256
#   ./run_resnet_profile.sh --batch 128      # custom batch size
#   ./run_resnet_profile.sh --report /tmp/r.json
#   ./run_resnet_profile.sh --no-kernel      # skip planned-node timing
#
# Outputs:
#   profile-resnet.json  — machine-readable profile
#   stderr               — human-readable summary table

PYTHON="${PYTHON:-python3}"
RELEASE=".specification/testcases/release_to_competitors"
MODEL="${RELEASE}/models/resnet_v1.onnx"
INPUT="${RELEASE}/testdata/c35/resnet_v1/input"
OUTPUT="/tmp/c35-profiled-resnet-$(date +%Y%m%d-%H%M%S)"
REPORT="${PROFILE_REPORT:-profile-resnet.json}"

BATCH_SIZE=256

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch|-b) BATCH_SIZE="$2"; shift 2 ;;
    --report|-r) REPORT="$2"; shift 2 ;;
    --output|-o) OUTPUT="$2"; shift 2 ;;
    --no-kernel) NO_KERNEL="--no-kernel-profile"; shift ;;
    --help|-h)
      sed -n '5,16p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -f "$MODEL" ]]   || { echo "Missing model: $MODEL" >&2; exit 2; }
[[ -d "$INPUT" ]]   || { echo "Missing input: $INPUT" >&2; exit 2; }

echo "=== C3.5 ResNet Profiler ===" >&2
echo "  Model:      $MODEL" >&2
echo "  Input:      $INPUT" >&2
echo "  Output:     $OUTPUT" >&2
echo "  Batch size: $BATCH_SIZE" >&2
echo "  Report:     $REPORT" >&2
echo "" >&2

exec "$PYTHON" -m c35.profiler \
  --onnx "$MODEL" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --batch-size "$BATCH_SIZE" \
  --report "$REPORT" \
  ${NO_KERNEL:-}
