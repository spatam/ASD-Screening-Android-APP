"""
Phase 3 - Knowledge Distillation into StudentASD.
==================================================
Distils the Phase 2 TwoStreamTeacher ensemble into a lightweight
StudentASD model via soft-label KD (temperature scaling).

Usage
-----
    python scripts/train_phase3.py \
        --huiyu-dir huiyu_2019_eye_movements \
        --phase2-runs-dir runs \
        --phase2-tag phase2_v1 \
        --output-dir runs \
        --run-tag distill_v1

Thin wrapper around ``asd_gaze.train_distill``.
"""

import subprocess, sys, argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 3: Knowledge distillation into StudentASD."
    )
    p.add_argument("--huiyu-dir",         required=True)
    p.add_argument("--phase2-runs-dir",   default="runs")
    p.add_argument("--phase2-tag",        default="phase2_v1")
    p.add_argument("--output-dir",        default="runs")
    p.add_argument("--run-tag",           default="distill_v1")
    p.add_argument("--epochs",            type=int,   default=80)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-2)
    p.add_argument("--warmup-epochs",     type=int,   default=5)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--max-seq-len",       type=int,   default=25)
    p.add_argument("--blend-alpha",       type=float, default=0.55)
    p.add_argument("--n-splits",          type=int,   default=5)
    p.add_argument("--temperature",       type=float, default=4.0)
    p.add_argument("--alpha",             type=float, default=0.7)
    p.add_argument("--student-hidden",    type=int,   default=64)
    args, extra = p.parse_known_args()

    ckpt_pattern = (
        f"{args.phase2_runs_dir}/best_fold{{fold}}_{args.phase2_tag}.pth"
    )

    cmd = [
        sys.executable, "-m", "asd_gaze.train_distill",
        "--huiyu-dir",            args.huiyu_dir,
        "--teacher-ckpt-pattern", ckpt_pattern,
        "--output-dir",           args.output_dir,
        "--run-tag",              args.run_tag,
        "--epochs",               str(args.epochs),
        "--batch-size",           str(args.batch_size),
        "--lr",                   str(args.lr),
        "--weight-decay",         str(args.weight_decay),
        "--warmup-epochs",        str(args.warmup_epochs),
        "--patience",             str(args.patience),
        "--max-seq-len",          str(args.max_seq_len),
        "--blend-alpha",          str(args.blend_alpha),
        "--n-splits",             str(args.n_splits),
        "--temperature",          str(args.temperature),
        "--alpha",                str(args.alpha),
        "--student-hidden",       str(args.student_hidden),
    ] + extra

    print("[Phase 3] Launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
