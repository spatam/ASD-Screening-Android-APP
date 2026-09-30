from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .student_model import StudentASD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export student checkpoints to ONNX.')
    parser.add_argument('--ckpt-pattern', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--student-hidden', type=int, default=64)
    parser.add_argument('--no-verify', action='store_true')
    return parser.parse_args()


def export_single(ckpt_path: Path, out_path: Path, opset: int, max_seq_len: int, hidden: int, verify: bool) -> None:
    device = torch.device('cpu')
    model = StudentASD(seq_feature_dim=3, hidden=hidden, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dummy_seq = torch.randn(1, max_seq_len, 3)
    dummy_visual = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        (dummy_seq, dummy_visual),
        out_path,
        dynamo=False,
        opset_version=opset,
        # Input names are the contract with OnnxEnsembleRunner in the Android app.
        input_names=['gaze_seq', 'stimulus_image'],
        output_names=['logit'],
        dynamic_axes={
            'gaze_seq': {0: 'batch', 1: 'seq_len'},
            'stimulus_image': {0: 'batch'},
            'logit': {0: 'batch'},
        },
        do_constant_folding=True,
    )

    if verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
        with torch.no_grad():
            pt_out = model(dummy_seq, dummy_visual).numpy()
        ort_out = session.run(['logit'], {'gaze_seq': dummy_seq.numpy(), 'stimulus_image': dummy_visual.numpy()})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        print(f'[ONNX] {out_path.name} max diff = {max_diff:.2e}')


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        ckpt_path = Path(args.ckpt_pattern.format(fold=fold))
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Student checkpoint not found: {ckpt_path}')
        out_path = output_dir / f'{ckpt_path.stem}.onnx'
        export_single(ckpt_path, out_path, args.opset, args.max_seq_len, args.student_hidden, verify=not args.no_verify)


if __name__ == '__main__':
    main()
