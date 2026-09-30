#!/usr/bin/env bash
# run_ablations.sh - Runs the paper's ablations (Section VI.A-VI.B).
#
# Usage:
#   bash training_scripts/run_ablations.sh --huiyu-dir <dir> [--he-dir <dir>] [--output-dir runs_ablation]
#
# Options:
#   --device        auto (default; auto/cpu/mps/cuda)
#   --python        interpreter to use
#   --only          comma-separated subset of:
#                   spatial_subject,image_subject,spatial_stimulus,image_stimulus,
#                   random_heatmap,ensemble,pretrain
#
# Dependency order:
#   spatial_subject  ->  random_heatmap, ensemble  (they reuse its checkpoints)
#   pretrain         ->  full_pretrained

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HUIYU_DIR=""
HE_DIR=""
OUTPUT_DIR="runs_ablation"
DEVICE="auto"
PYTHON="${PYTHON:-python3}"
ONLY="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --huiyu-dir)  HUIYU_DIR="$2"; shift 2 ;;
        --he-dir)     HE_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --python)     PYTHON="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        *) echo "[run_ablations] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$HUIYU_DIR" ]]; then
    echo "[run_ablations] ERROR: --huiyu-dir is required." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

wants() {
    [[ "$ONLY" == "all" ]] && return 0
    [[ ",$ONLY," == *",$1,"* ]]
}

_elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dh%02dm" $(( secs/3600 )) $(( (secs%3600)/60 ))
}

banner() {
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo " $1"
    echo "──────────────────────────────────────────────────────"
}

train_one() {
    local ablation="$1" protocol="$2"
    local t0; t0=$(date +%s)
    banner "$ablation / $protocol"
    "$PYTHON" -u -m asd_gaze.train_ablation \
        --huiyu-dir "$HUIYU_DIR" \
        --ablation "$ablation" --protocol "$protocol" \
        --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    echo "[$ablation/$protocol] done in $(_elapsed $t0)"
}

wants spatial_subject   && train_one spatial_only subject
wants image_subject     && train_one image_only   subject
wants spatial_stimulus  && train_one spatial_only stimulus
wants image_stimulus    && train_one image_only   stimulus

# Random-heatmap control. It reuses the spatial-only checkpoints and trains nothing.
if wants random_heatmap; then
    banner "random-heatmap (evaluation only)"
    "$PYTHON" -u -m asd_gaze.eval_ablation \
        --huiyu-dir "$HUIYU_DIR" --mode random_heatmap \
        --ckpt-dir "$OUTPUT_DIR" --ckpt-tag spatial_only_subject \
        --output-dir "$OUTPUT_DIR" --device "$DEVICE"
fi

# Selection-bias analysis with a 5-checkpoint ensemble on the retained and discarded subsets.
if wants ensemble; then
    banner "5-checkpoint ensemble (retained vs discarded)"
    for retention in retained discarded; do
        "$PYTHON" -u -m asd_gaze.eval_ablation \
            --huiyu-dir "$HUIYU_DIR" --mode ensemble --retention "$retention" \
            --ckpt-dir "$OUTPUT_DIR" --ckpt-tag spatial_only_subject \
            --output-dir "$OUTPUT_DIR" --device "$DEVICE"
    done
fi

# Cross-dataset pre-training of the temporal branch, then a full teacher that uses it.
if wants pretrain; then
    if [[ -z "$HE_DIR" ]]; then
        echo "[run_ablations] --he-dir not given. Skipping the pre-training ablation." >&2
    else
        banner "cross-dataset temporal pre-training"
        "$PYTHON" -u -m asd_gaze.pretrain_temporal \
            --he-dir "$HE_DIR" --out "$OUTPUT_DIR/temporal_pretrained.pth" --device "$DEVICE"
        "$PYTHON" -u -m asd_gaze.train_ablation \
            --huiyu-dir "$HUIYU_DIR" --ablation full --protocol subject \
            --temporal-ckpt "$OUTPUT_DIR/temporal_pretrained.pth" \
            --output-dir "$OUTPUT_DIR" --run-tag full_pretrained --device "$DEVICE"
    fi
fi

banner "Ablations complete. Results in: $OUTPUT_DIR/"
ls -1 "$OUTPUT_DIR"/summary_*.csv "$OUTPUT_DIR"/eval_*.csv 2>/dev/null || true
