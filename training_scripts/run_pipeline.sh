#!/usr/bin/env bash
# run_pipeline.sh – End-to-end 4-phase ASD gaze pipeline orchestrator.
#
# Usage:
#   bash training_scripts/run_pipeline.sh --huiyu-dir /path/to/huiyu_2019_eye_movements
#
# Optional overrides:
#   --output-dir   runs          (default)
#   --phase1-tag   phase1_v1     (default)
#   --phase2-tag   phase2_v1     (default)
#   --phase3-tag   distill_v1    (default)
#   --device       auto          (default; auto/cpu/mps/cuda)
#   --python       python3       (default; override interpreter path)
#   --progress-file <output-dir>/progress.md  (default; one entry per finished phase)
#   --gpu          0             (CUDA device; set to -1 for CPU)
#   --protocol     subject       (subject: original 2216-record protocol;
#                                 stimulus: held-out-stimulus, 7296 records,
#                                 folds grouped by image identity)
#   --skip-phase1                (resume from existing Phase 1 checkpoints)
#   --skip-phase2
#   --skip-phase3
#   --skip-phase4
#   --skip-eval
#
# Run from the project root directory:
#   cd /path/to/android-app && bash training_scripts/run_pipeline.sh --huiyu-dir /path/to/huiyu_2019_eye_movements

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── defaults ────────────────────────────────────────────────────────────────
HUIYU_DIR=""
OUTPUT_DIR="runs"
PHASE1_TAG="phase1_v1"
PHASE2_TAG="phase2_v1"
PHASE3_TAG="distill_v1"
DEVICE="auto"
PYTHON_OVERRIDE=""
PROGRESS_FILE=""
GPU="0"
# Cross-validation protocol passed to every phase and to the evaluation.
# Empty = the code default (subject), so earlier runs stay unchanged.
PROTOCOL=""
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
        --device)      DEVICE="$2"; shift 2 ;;
        --python)      PYTHON_OVERRIDE="$2"; shift 2 ;;
        --progress-file) PROGRESS_FILE="$2"; shift 2 ;;
        --gpu)         GPU="$2"; shift 2 ;;
        --protocol)    PROTOCOL="$2"; shift 2 ;;
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

PROGRESS_FILE="${PROGRESS_FILE:-$OUTPUT_DIR/progress.md}"

# ─── environment ─────────────────────────────────────────────────────────────
# Activate .venv if present; otherwise rely on the environment's Python.
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

PYTHON="${PYTHON_OVERRIDE:-${PYTHON:-python3}}"

# Unbuffered log. Without it, per-epoch progress stays in the stdout buffer and
# the log file looks stuck even while training runs.
export PYTHONUNBUFFERED=1

# Set CUDA device (ignored on Apple Silicon / CPU-only machines).
if [[ "$GPU" != "-1" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "========================================================"
echo " ASD Gaze Pipeline  –  4-phase"
echo "  huiyu-dir  : $HUIYU_DIR"
echo "  output-dir : $OUTPUT_DIR"
echo "  tags       : $PHASE1_TAG / $PHASE2_TAG / $PHASE3_TAG"
echo "  device     : $DEVICE"
echo "  python     : $PYTHON"
echo "  GPU        : ${GPU}"
echo "  protocol   : ${PROTOCOL:-subject (default)}"
echo "========================================================"

# Only the phases that train or evaluate get this. The ONNX export does not.
PROTOCOL_ARGS=()
if [[ -n "$PROTOCOL" ]]; then
    PROTOCOL_ARGS=(--protocol "$PROTOCOL")
fi

_elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dm%02ds" $(( secs/60 )) $(( secs%60 ))
}

append_progress() {
    local phase="$1"
    local details="$2"
    mkdir -p "$(dirname "$PROGRESS_FILE")"
    {
        echo ""
        echo "## $(date '+%Y-%m-%d %H:%M:%S') - $phase"
        echo ""
        echo "- status: completed"
        echo "- output_dir: $OUTPUT_DIR"
        echo "- details: $details"
    } >> "$PROGRESS_FILE"
}

# ─── Phase 1 ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " Phase 1: Two-Stream Teacher (random init)"
    echo "──────────────────────────────────────────────────────"
    T0=$(date +%s)
    "$PYTHON" training_scripts/train_phase1.py \
        --huiyu-dir  "$HUIYU_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --run-tag    "$PHASE1_TAG" \
        --device     "$DEVICE" "${PROTOCOL_ARGS[@]}"
    echo "[Phase 1] done in $(_elapsed $T0)"
    append_progress "Phase 1" "Teacher trained with tag $PHASE1_TAG"
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
    "$PYTHON" training_scripts/train_phase2.py \
        --huiyu-dir          "$HUIYU_DIR" \
        --phase1-runs-dir    "$OUTPUT_DIR" \
        --phase1-tag         "$PHASE1_TAG" \
        --output-dir         "$OUTPUT_DIR" \
        --run-tag            "$PHASE2_TAG" \
        --device             "$DEVICE" "${PROTOCOL_ARGS[@]}"
    echo "[Phase 2] done in $(_elapsed $T0)"
    append_progress "Phase 2" "Augmentation-regularised teacher trained with tag $PHASE2_TAG"
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
    "$PYTHON" training_scripts/train_phase3.py \
        --huiyu-dir          "$HUIYU_DIR" \
        --phase2-runs-dir    "$OUTPUT_DIR" \
        --phase2-tag         "$PHASE2_TAG" \
        --output-dir         "$OUTPUT_DIR" \
        --run-tag            "$PHASE3_TAG" \
        --device             "$DEVICE" "${PROTOCOL_ARGS[@]}"
    echo "[Phase 3] done in $(_elapsed $T0)"
    append_progress "Phase 3" "Student distilled with tag $PHASE3_TAG"
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
    "$PYTHON" training_scripts/export_onnx.py \
        --phase3-runs-dir    "$OUTPUT_DIR" \
        --phase3-tag         "$PHASE3_TAG" \
        --output-dir         "$OUTPUT_DIR/onnx"
    echo "[Phase 4] done in $(_elapsed $T0)"
    append_progress "Phase 4" "ONNX checkpoints exported to $OUTPUT_DIR/onnx"
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
    "$PYTHON" training_scripts/evaluate.py \
        --huiyu-dir  "$HUIYU_DIR" \
        --runs-dir   "$OUTPUT_DIR" \
        --phase1-tag "$PHASE1_TAG" \
        --phase2-tag "$PHASE2_TAG" \
        --phase3-tag "$PHASE3_TAG" \
        --device     "$DEVICE" "${PROTOCOL_ARGS[@]}"
    echo "[Evaluate] done in $(_elapsed $T0)"
    append_progress "Evaluation" "Metrics written under $OUTPUT_DIR"
else
    echo "[Evaluate] skipped."
fi

echo ""
echo "========================================================"
echo " Pipeline complete."
echo " Results in: $OUTPUT_DIR/"
echo "========================================================"
