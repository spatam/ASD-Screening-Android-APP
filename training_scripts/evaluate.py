"""
Evaluation - full per-fold and summary metrics.
================================================
Runs OOF evaluation across all phases and writes two CSV files:
  * runs/full_metrics_per_fold.csv
  * runs/full_metrics_summary.csv

Usage
-----
    python scripts/evaluate.py \
        --huiyu-dir huiyu_2019_eye_movements \
        --runs-dir runs \
        --phase1-tag phase1_v1 \
        --phase2-tag phase2_v1 \
        --phase3-tag distill_v1

Thin wrapper around ``asd_gaze.compute_full_metrics``.
"""

import subprocess, sys, argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute full per-fold metrics for all phases."
    )
    p.add_argument("--huiyu-dir",    required=True)
    p.add_argument("--runs-dir",     default="runs")
    p.add_argument("--phase1-tag",   default="phase1_v1")
    p.add_argument("--phase2-tag",   default="phase2_v1")
    p.add_argument("--phase3-tag",   default="distill_v1")
    p.add_argument("--n-splits",     type=int,   default=5)
    p.add_argument("--max-seq-len",  type=int,   default=25)
    p.add_argument("--blend-alpha",  type=float, default=0.55)
    args, extra = p.parse_known_args()

    cmd = [
        sys.executable, "-m", "asd_gaze.compute_full_metrics",
        "--huiyu-dir",   args.huiyu_dir,
        "--runs-dir",    args.runs_dir,
        "--phase1-tag",  args.phase1_tag,
        "--phase2-tag",  args.phase2_tag,
        "--phase3-tag",  args.phase3_tag,
        "--n-splits",    str(args.n_splits),
        "--max-seq-len", str(args.max_seq_len),
        "--blend-alpha", str(args.blend_alpha),
    ] + extra

    print("[Evaluate] Launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
