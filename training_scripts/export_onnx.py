"""
Phase 4 - Export StudentASD checkpoints to ONNX.
=================================================
Iterates over all per-fold distillation checkpoints and exports each
one to ONNX (opset 18) for on-device inference.

Usage
-----
    python scripts/export_onnx.py \
        --phase3-runs-dir runs \
        --phase3-tag distill_v1 \
        --output-dir runs/onnx

Thin wrapper around ``asd_gaze.export_onnx``.
"""

import subprocess, sys, argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 4: Export student checkpoints to ONNX."
    )
    p.add_argument("--phase3-runs-dir",  default="runs")
    p.add_argument("--phase3-tag",       default="distill_v1")
    p.add_argument("--output-dir",       default="runs/onnx")
    p.add_argument("--folds",            type=int, default=5)
    p.add_argument("--opset",            type=int, default=18)
    p.add_argument("--max-seq-len",      type=int, default=25)
    p.add_argument("--student-hidden",   type=int, default=64)
    p.add_argument("--no-verify",        action="store_true",
                   help="Skip ONNX runtime verification after export.")
    args, extra = p.parse_known_args()

    ckpt_pattern = (
        f"{args.phase3_runs_dir}/best_fold{{fold}}_{args.phase3_tag}.pth"
    )

    cmd = [
        sys.executable, "-m", "asd_gaze.export_onnx",
        "--ckpt-pattern",    ckpt_pattern,
        "--output-dir",      args.output_dir,
        "--folds",           str(args.folds),
        "--opset",           str(args.opset),
        "--max-seq-len",     str(args.max_seq_len),
        "--student-hidden",  str(args.student_hidden),
    ]
    if args.no_verify:
        cmd.append("--no-verify")
    cmd += extra

    print("[Phase 4] Launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
