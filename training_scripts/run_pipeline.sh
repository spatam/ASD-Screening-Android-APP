#!/usr/bin/env bash
# run_pipeline.sh – End-to-end 4-phase ASD gaze pipeline orchestrator.
#
# Usage:
#   bash scripts/run_pipeline.sh --huiyu-dir huiyu_2019_eye_movements
#
# Optional overrides:
#   --output-dir   runs          (default)
#   --phase1-tag   phase1_v1     (default)
#   --phase2-tag   phase2_v1     (default)
#   --phase3-tag   distill_v1    (default)
#   --gpu          0             (CUDA device; set to -1 for CPU)
#   --skip-phase1                (resume from existing Phase 1 checkpoints)
#   --skip-phase2
#   --skip-phase3
#   --skip-phase4
#   --skip-eval
#
# Run from the project root directory:
#   cd /path/to/max_project && bash scripts/run_pipeline.sh --huiyu-dir huiyu_2019_eye_movements

set -euo pipefail

# ─── defaults ────────────────────────────────────────────────────────────────
HUIYU_DIR=""
OUTPUT_DIR="runs"
PHASE1_TAG="phase1_v1"
PHASE2_TAG="phase2_v1"
PHASE3_TAG="distill_v1"
GPU="0"
SKIP_PHASE1=0
SKIP_PHASE2=0
SKIP_PHASE3=0
SKIP_PHASE4=0
SKIP_EVAL=0

# ─── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --huiyu-dir)   HUIYU_DIR="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --phase1-tag)  PHASE1_TAG="$2"; shift 2 ;;
        --phase2-tag)  PHASE2_TAG="$2"; shift 2 ;;
        --phase3-tag)  PHASE3_TAG="$2"; shift 2 ;;
        --gpu)         GPU="$2"; shift 2 ;;
        --skip-phase1) SKIP_PHASE1=1; shift ;;
        --skip-phase2) SKIP_PHASE2=1; shift ;;
        --skip-phase3) SKIP_PHASE3=1; shift ;;
        --skip-phase4) SKIP_PHASE4=1; shift ;;
        --skip-eval)   SKIP_EVAL=1; shift ;;
        *) echo "[run_pipeline] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$HUIYU_DIR" ]]; then
    echo "[run_pipeline] ERROR: --huiyu-dir is required." >&2
    exit 1
fi

# ─── environment ─────────────────────────────────────────────────────────────
# Activate .venv if present; otherwise rely on the environment's Python.
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

PYTHON="${PYTHON:-python3}"

# Set CUDA device (ignored on CPU-only machines).
if [[ "$GPU" != "-1" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "========================================================"
echo " ASD Gaze Pipeline  –  4-phase"
echo "  huiyu-dir  : $HUIYU_DIR"
echo "  output-dir : $OUTPUT_DIR"
echo "  tags       : $PHASE1_TAG / $PHASE2_TAG / $PHASE3_TAG"
echo "  GPU        : ${GPU}"
echo "========================================================"

_elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dm%02ds" $(( secs/60 )) $(( secs%60 ))
}

# ─── Phase 1 ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Phase 1: Two-Stream Teacher (random init)"
    echo "──────────────────────────────────────────────────────"
    T0=$(date +%s)
    "$PYTHON" scripts/train_phase1.py \
        --huiyu-dir  "$HUIYU_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --run-tag    "$PHASE1_TAG"
    echo "[Phase 1] done in $(_elapsed $T0)"
else
    echo "[Phase 1] skipped."
fi

# ─── Phase 2 ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Phase 2: Teacher fine-tuning (focal loss + mixup)"
    echo "──────────────────────────────────────────────────────"
    T0=$(date +%s)
    "$PYTHON" scripts/train_phase2.py \
        --huiyu-dir          "$HUIYU_DIR" \
        --phase1-runs-dir    "$OUTPUT_DIR" \
        --phase1-tag         "$PHASE1_TAG" \
        --output-dir         "$OUTPUT_DIR" \
        --run-tag            "$PHASE2_TAG"
    echo "[Phase 2] done in $(_elapsed $T0)"
else
    echo "[Phase 2] skipped."
fi

# ─── Phase 3 ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_PHASE3" -eq 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Phase 3: Knowledge Distillation → StudentASD"
    echo "──────────────────────────────────────────────────────"
    T0=$(date +%s)
    "$PYTHON" scripts/train_phase3.py \
        --huiyu-dir          "$HUIYU_DIR" \
        --phase2-runs-dir    "$OUTPUT_DIR" \
        --phase2-tag         "$PHASE2_TAG" \
        --output-dir         "$OUTPUT_DIR" \
        --run-tag            "$PHASE3_TAG"
    echo "[Phase 3] done in $(_elapsed $T0)"
else
    echo "[Phase 3] skipped."
fi

# ─── Phase 4 ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_PHASE4" -eq 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Phase 4: ONNX export (opset 18)"
    echo "──────────────────────────────────────────────────────"
    T0=$(date +%s)
    "$PYTHON" scripts/export_onnx.py \
        --phase3-runs-dir    "$OUTPUT_DIR" \
        --phase3-tag         "$PHASE3_TAG" \
        --output-dir         "$OUTPUT_DIR/onnx"
    echo "[Phase 4] done in $(_elapsed $T0)"
else
    echo "[Phase 4] skipped."
fi

# ─── Evaluation ──────────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" -eq 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Evaluation: per-fold & summary metrics"
    echo "──────────────────────────────────────────────────────"
    T0=$(date +%s)
    "$PYTHON" scripts/evaluate.py \
        --huiyu-dir  "$HUIYU_DIR" \
        --runs-dir   "$OUTPUT_DIR" \
        --phase1-tag "$PHASE1_TAG" \
        --phase2-tag "$PHASE2_TAG" \
        --phase3-tag "$PHASE3_TAG"
    echo "[Evaluate] done in $(_elapsed $T0)"
else
    echo "[Evaluate] skipped."
fi

echo ""
echo "========================================================"
echo " Pipeline complete."
echo " Results in: $OUTPUT_DIR/"
echo "========================================================"
