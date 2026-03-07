"""
Phase 1 – Two-Stream Teacher (random initialisation).
======================================================
Trains TwoStreamTeacher on Huiyu 2019 per-image records under a
5-fold Stratified Group CV (grouped by subject).

Usage
-----
    python scripts/train_phase1.py \
        --huiyu-dir huiyu_2019_eye_movements \
        --output-dir runs \
        --run-tag phase1_v1

The script is a thin wrapper around ``asd_gaze.train_phase2`` with
``--temporal-ckpt ""`` (random initialisation, no pre-training).
"""

import subprocess
import sys
import argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 1: Two-Stream Teacher from random initialisation."
    )
    p.add_argument("--huiyu-dir",        required=True,
                   help="Path to huiyu_2019_eye_movements directory.")
    p.add_argument("--output-dir",       default="runs")
    p.add_argument("--run-tag",          default="phase1_v1")
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lr-temporal",      type=float, default=1e-5)
    p.add_argument("--weight-decay",     type=float, default=1e-2)
    p.add_argument("--warmup-epochs",    type=int,   default=5)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--max-seq-len",      type=int,   default=25)
    p.add_argument("--blend-alpha",      type=float, default=0.55)
    p.add_argument("--n-splits",         type=int,   default=5)
    args, extra = p.parse_known_args()

    cmd = [
        sys.executable, "-m", "asd_gaze.train_phase2",
        "--huiyu-dir",       args.huiyu_dir,
        "--output-dir",      args.output_dir,
        "--run-tag",         args.run_tag,
        "--epochs",          str(args.epochs),
        "--batch-size",      str(args.batch_size),
        "--lr",              str(args.lr),
        "--lr-temporal",     str(args.lr_temporal),
        "--weight-decay",    str(args.weight_decay),
        "--warmup-epochs",   str(args.warmup_epochs),
        "--patience",        str(args.patience),
        "--max-seq-len",     str(args.max_seq_len),
        "--blend-alpha",     str(args.blend_alpha),
        "--n-splits",        str(args.n_splits),
        "--temporal-ckpt",   "",   # no pre-training → random init
    ] + extra

    print("[Phase 1] Launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
