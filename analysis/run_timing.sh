#!/usr/bin/env bash
# Instrumented re-run of Phases 1-3 that records wall-clock time and peak memory with
# /usr/bin/time. The paper's efficiency and cost tables use these numbers (Apple M4 Pro, MPS,
# retained-subject protocol).
#
# Usage:
#   bash analysis/run_timing.sh /path/to/saliency4asd [output_dir] [device]
set -uo pipefail

HUIYU="${1:?usage: run_timing.sh /path/to/saliency4asd [output_dir] [device]}"
OUT="${2:-runs_timing}"
DEVICE="${3:-auto}"
PY="${PYTHON:-python3}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

# macOS reports peak memory with -l, GNU time with -v.
if /usr/bin/time -l true >/dev/null 2>&1; then TIME_FLAG=-l; else TIME_FLAG=-v; fi

run_phase() {
    local name="$1"; shift
    echo "=== $name start $(date '+%Y-%m-%d %H:%M:%S') ==="
    /usr/bin/time "$TIME_FLAG" "$PY" "$@" > "$OUT/timing_$name.log" 2>&1
    echo "=== $name end   $(date '+%Y-%m-%d %H:%M:%S') ==="
}

run_phase phase1 training_scripts/train_phase1.py \
    --huiyu-dir "$HUIYU" --output-dir "$OUT" --run-tag timing_phase1 --device "$DEVICE"
run_phase phase2 training_scripts/train_phase2.py \
    --huiyu-dir "$HUIYU" --phase1-runs-dir "$OUT" --phase1-tag timing_phase1 \
    --output-dir "$OUT" --run-tag timing_phase2 --device "$DEVICE"
run_phase phase3 training_scripts/train_phase3.py \
    --huiyu-dir "$HUIYU" --phase2-runs-dir "$OUT" --phase2-tag timing_phase2 \
    --output-dir "$OUT" --run-tag timing_distill --device "$DEVICE"

echo "TIMING_RUN_COMPLETE"
