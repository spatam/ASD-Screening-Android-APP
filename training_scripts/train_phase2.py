"""
Phase 2 - Two-Stream Teacher with augmentation.
================================================
Fine-tunes TwoStreamTeacher (pre-trained in Phase 1) using focal loss,
label smoothing, and mixup augmentation.

Usage
-----
    python scripts/train_phase2.py \
        --huiyu-dir huiyu_2019_eye_movements \
        --phase1-runs-dir runs \
        --phase1-tag phase1_v1 \
        --output-dir runs \
        --run-tag phase2_v1

Thin wrapper around ``asd_gaze.train_phase3``.
"""

import subprocess, sys, argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 2: Teacher fine-tuning with focal loss + mixup."
    )
    p.add_argument("--huiyu-dir",         required=True)
    p.add_argument("--phase1-runs-dir",   default="runs")
    p.add_argument("--phase1-tag",        default="phase1_v1")
    p.add_argument("--output-dir",        default="runs")
    p.add_argument("--run-tag",           default="phase2_v1")
    p.add_argument("--epochs",            type=int,   default=80)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--lr-temporal",       type=float, default=1e-5)
    p.add_argument("--weight-decay",      type=float, default=1e-2)
    p.add_argument("--warmup-epochs",     type=int,   default=5)
    p.add_argument("--patience",          type=int,   default=25)
    p.add_argument("--max-seq-len",       type=int,   default=25)
    p.add_argument("--blend-alpha",       type=float, default=0.55)
    p.add_argument("--n-splits",          type=int,   default=5)
    p.add_argument("--label-smoothing",   type=float, default=0.1)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--mixup-alpha",       type=float, default=0.2)
    p.add_argument("--gaze-jitter",       type=float, default=0.02)
    p.add_argument("--fix-dropout",       type=float, default=0.1)
    args, extra = p.parse_known_args()

    ckpt_pattern = (
        f"{args.phase1_runs_dir}/best_fold{{fold}}_{args.phase1_tag}.pth"
    )

    cmd = [
        sys.executable, "-m", "asd_gaze.train_phase3",
        "--huiyu-dir",        args.huiyu_dir,
        "--temporal-ckpt",    ckpt_pattern,
        "--output-dir",       args.output_dir,
        "--run-tag",          args.run_tag,
        "--epochs",           str(args.epochs),
        "--batch-size",       str(args.batch_size),
        "--lr",               str(args.lr),
        "--lr-temporal",      str(args.lr_temporal),
        "--weight-decay",     str(args.weight_decay),
        "--warmup-epochs",    str(args.warmup_epochs),
        "--patience",         str(args.patience),
        "--max-seq-len",      str(args.max_seq_len),
        "--blend-alpha",      str(args.blend_alpha),
        "--n-splits",         str(args.n_splits),
        "--label-smoothing",  str(args.label_smoothing),
        "--focal-gamma",      str(args.focal_gamma),
        "--mixup-alpha",      str(args.mixup_alpha),
        "--gaze-jitter",      str(args.gaze_jitter),
        "--fix-dropout",      str(args.fix_dropout),
    ] + extra

    print("[Phase 2] Launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
