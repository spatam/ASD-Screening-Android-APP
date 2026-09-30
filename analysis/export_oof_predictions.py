"""Export per-record out-of-fold predictions with their (subject, image) identity.

``compute_full_metrics`` computes only aggregate metrics. It uses the OOF
probabilities and discards them. In real use the app shows K images to a new
child and combines the K predictions into one subject-level decision.
Simulating this needs the individual probabilities, each tagged with the
subject and the image that produced it.

Each record gets its prediction from the checkpoint of the fold that had its
subject in validation, so the exported probabilities are truly out-of-fold.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from asd_gaze.model import TwoStreamTeacher
from asd_gaze.student_model import StudentASD
from asd_gaze.train_common import add_protocol_argument, build_protocol_dataset, make_loader
from asd_gaze.utils import detect_device


PHASE_BUILDERS = {
    'phase1': lambda: TwoStreamTeacher(seq_feature_dim=3, pretrained=False),
    'phase2': lambda: TwoStreamTeacher(seq_feature_dim=3, pretrained=False),
    'phase3': lambda: StudentASD(seq_feature_dim=3, pretrained=False),
}


@torch.no_grad()
def collect_oof(dataset, labels, groups, runs_dir: Path, tag: str, phase: str,
                n_splits: int, device: torch.device, batch_size: int):
    """OOF probability and validation-fold index for each record."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.full(len(dataset), np.nan, dtype=np.float64)
    fold_of = np.full(len(dataset), -1, dtype=np.int32)

    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(dataset)), labels, groups)):
        ckpt = runs_dir / f'best_fold{fold}_{tag}.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt}')
        model = PHASE_BUILDERS[phase]()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()

        # shuffle=False keeps the predictions in val_idx order.
        loader = make_loader(dataset.with_indices(val_idx), batch_size=batch_size, shuffle=False)
        chunks = []
        for seq, visual, _, _ in loader:
            out = model(seq.to(device), visual.to(device))
            chunks.append(torch.sigmoid(out).detach().cpu().numpy().reshape(-1))
        fold_probs = np.concatenate(chunks) if chunks else np.array([])
        assert len(fold_probs) == len(val_idx), (
            f'{phase} fold{fold}: {len(fold_probs)} predictions for {len(val_idx)} records'
        )
        probs[val_idx] = fold_probs
        fold_of[val_idx] = fold
        del model

    assert not np.isnan(probs).any(), 'Some records did not get an OOF prediction'
    return probs, fold_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--runs-dir', required=True)
    parser.add_argument('--phase1-tag', default='phase1_v1')
    parser.add_argument('--phase2-tag', default='phase2_v1')
    parser.add_argument('--phase3-tag', default='distill_v1')
    parser.add_argument('--phases', default='phase1,phase2,phase3',
                        help='Comma-separated phases to export.')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--blend-alpha', type=float, default=0.55)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', required=True, help='Output CSV.')
    add_protocol_argument(parser)
    args = parser.parse_args()

    device = detect_device(args.device)
    dataset = build_protocol_dataset(args)
    groups, labels = dataset.get_groups_and_labels()
    print(f'device={device} | records={len(dataset)} | subjects={len(set(groups))}')

    tags = {'phase1': args.phase1_tag, 'phase2': args.phase2_tag, 'phase3': args.phase3_tag}
    wanted = [p.strip() for p in args.phases.split(',') if p.strip()]

    results: dict[str, np.ndarray] = {}
    fold_of = None
    for phase in wanted:
        print(f'--- {phase} (tag={tags[phase]}) ---')
        probs, folds = collect_oof(dataset, labels, groups, Path(args.runs_dir),
                                   tags[phase], phase, args.n_splits, device, args.batch_size)
        results[phase] = probs
        fold_of = folds if fold_of is None else fold_of

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ['record_idx', 'image_id', 'subject', 'slot', 'label', 'fold'] + [f'prob_{p}' for p in wanted]
    with out_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, record in enumerate(dataset.records):
            row = [i, record.image_id, record.group, record.slot, record.label, int(fold_of[i])]
            row += [f'{results[p][i]:.6f}' for p in wanted]
            writer.writerow(row)
    print(f'Wrote {out_path} ({len(dataset.records)} rows)')


if __name__ == '__main__':
    main()
