#!/usr/bin/env bash
# Build and verify a clean submission archive.
#
# Usage:
#   bash scripts/build_submission.sh [OUTPUT]
#
# This script:
#   1. Exports the tracked files in HEAD (`export-ignore` attributes apply;
#      untracked and ignored workspace files are never included).
#   2. Lists every member and checks for forbidden patterns.
#   3. Reports pass/fail with a summary.
#
# The resulting archive is the submission artifact.  The verification step
# is reproducible evidence that the archive is clean — `.gitignore` alone is
# packaging policy, not proof.

set -euo pipefail

OUTPUT="${1:-submission.tar.gz}"
FORBIDDEN_PATTERNS=(
    '__pycache__/'
    '\.pyc$'
    '\.pyo$'
    '\.venv/'
    'venv/'
    'env/'
    '\.whl$'
    '\.dag\.json$'
    '-report\.json$'
    '\.plan\.json$'
    '\.cache$'
    '\.log$'
    '\.DS_Store$'
    'Thumbs\.db$'
    '\.swp$'
    '\.swo$'
    '\.ssh/'
    '\.agents/'
    '\.specification/'
    '\.vscode/'
    '\.idea/'
    'output/'
    '\.git/'
)

PASS=0
FAIL=0
VIOLATIONS=()

echo "=== Building submission archive: $OUTPUT ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Export clean git archive
# ---------------------------------------------------------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: Not inside a git repository. Aborting."
    exit 1
fi

git archive --format=tar.gz --output="$OUTPUT" HEAD
echo "Archive created: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo ""

# ---------------------------------------------------------------------------
# 2. List members and check for forbidden patterns
# ---------------------------------------------------------------------------
echo "=== Archive member list ==="
MEMBERS=$(tar tzf "$OUTPUT")
echo "$MEMBERS"
echo ""

MEMBER_COUNT=$(echo "$MEMBERS" | wc -l | tr -d ' ')
echo "Total members: $MEMBER_COUNT"
echo ""

echo "=== Forbidden-pattern scan ==="
while IFS= read -r member; do
    for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
        if echo "$member" | grep -Eq "$pattern"; then
            echo "  VIOLATION: $member  (matches '$pattern')"
            VIOLATIONS+=("$member")
            ((FAIL++)) || true
        fi
    done
done <<< "$MEMBERS"

if [ ${#VIOLATIONS[@]} -eq 0 ]; then
    echo "  No forbidden patterns found."
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Expected-files sanity check
# ---------------------------------------------------------------------------
echo "=== Required-entry check ==="
REQUIRED=(
    'c31/__init__.py'
    'c31/import_onnx.py'
    'c32/__init__.py'
    'c33/__init__.py'
    'c34/__init__.py'
    'c35/__init__.py'
    'c35/deploy.py'
    'c35/standard_runner.py'
    'c3common/__init__.py'
    'README.md'
    'run_c35.sh'
)

for required in "${REQUIRED[@]}"; do
    if echo "$MEMBERS" | grep -Fxq "$required"; then
        echo "  OK: $required"
        ((PASS++)) || true
    else
        echo "  MISSING: $required"
        ((FAIL++)) || true
    fi
done

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Result ==="
echo "  Passes:  $PASS"
echo "  Failures: $FAIL"
echo "  Archive: $OUTPUT"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "ARCHIVE CHECK FAILED — do not submit until violations are resolved."
    exit 1
else
    echo ""
    echo "ARCHIVE CHECK PASSED — $OUTPUT is clean and ready for submission."
fi
