"""Evaluation-only ablations, with no training.

Two controls in the paper train no new models. Both reuse the spatial-only
checkpoints:

``random_heatmap``
    Out-of-fold evaluation that replaces every real heatmap with a random
    surrogate drawn from the global fixation distribution. It uses the same
    split as the spatial-only training, so the checkpoint that scores each
    record never saw it. The paper reports OOF AUC 0.9820 and PR-AUC 0.9724
    (the CV AUC column is empty because there is no training).

``ensemble``
    Inference on the whole chosen subset, averaging the 5 checkpoints. It
    supports the selection-bias analysis. In the paper the ``retained`` subset
    scores AUC 0.9852 and the ``discarded`` subset AUC 0.2575.

Typical use:

    python -m asd_gaze.eval_ablation --huiyu-dir <dir> --mode random_heatmap \
        --ckpt-dir runs_ablation --ckpt-tag spatial_only_subject
    python -m asd_gaze.eval_ablation --huiyu-dir <dir> --mode ensemble \
        --retention discarded --ckpt-dir runs_ablation --ckpt-tag spatial_only_subject
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .ablation_models import SpatialOnlyTeacher
from .dataset import HuiyuPerImageDataset
from .train_common import auc_pr, make_loader, predict_probabilities, write_rows
from .utils import detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluation-only ablations (no training).')
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--mode', required=True, choices=['random_heatmap', 'ensemble'])
    parser.add_argument('--ckpt-dir', default='runs_ablation')
    parser.add_argument('--ckpt-tag', default='spatial_only_subject')
    parser.add_argument('--retention', default='discarded', choices=['retained', 'discarded', 'all'],
                        help="Subset evaluated in ensemble mode.")
    parser.add_argument('--input-mode', default='blend', choices=['blend', 'image_only', 'random_heatmap'],
                        help="Input used in ensemble mode.")
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--blend-alpha', type=float, default=0.55)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit-records', type=int, default=0, help='>0 to truncate the dataset (smoke test only)')
    return parser.parse_args()


def load_checkpoint(ckpt_dir: Path, tag: str, fold: int, device: torch.device) -> SpatialOnlyTeacher:
    path = ckpt_dir / f'best_fold{fold}_{tag}.pth'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing checkpoint: {path}. Run the spatial-only training first with '
            f'`python -m asd_gaze.train_ablation --ablation spatial_only`.'
        )
    model = SpatialOnlyTeacher(seq_feature_dim=3, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_random_heatmap(args: argparse.Namespace, device: torch.device) -> dict:
    """OOF with surrogate heatmaps, on the same split used for spatial-only."""
    dataset = HuiyuPerImageDataset(
        args.huiyu_dir,
        max_seq_len=args.max_seq_len,
        blend_alpha=args.blend_alpha,
        input_mode='random_heatmap',
        retention='retained',
        group_by='subject',
    )
    groups, labels = dataset.get_groups_and_labels()
    if args.limit_records:
        keep = np.linspace(0, len(dataset) - 1, num=min(args.limit_records, len(dataset)), dtype=int)
        dataset.records = [dataset.records[i] for i in keep]
        groups, labels = dataset.get_groups_and_labels()

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(dataset), dtype=np.float32)
    ckpt_dir = Path(args.ckpt_dir)

    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(dataset)), labels, groups)):
        model = load_checkpoint(ckpt_dir, args.ckpt_tag, fold, device)
        loader = make_loader(dataset.with_indices(val_idx), batch_size=args.batch_size, shuffle=False)
        oof[val_idx] = predict_probabilities(model, loader, device)
        print(f'  fold {fold}: {len(val_idx)} records evaluated')

    auc, pr = auc_pr(labels, oof)
    return {'mode': 'random_heatmap', 'retention': 'retained', 'input_mode': 'random_heatmap',
            'n_records': len(dataset), 'auc': round(auc, 6), 'pr_auc': round(pr, 6)}


def run_ensemble(args: argparse.Namespace, device: torch.device) -> dict:
    """Mean probability of the 5 checkpoints over the whole subset."""
    dataset = HuiyuPerImageDataset(
        args.huiyu_dir,
        max_seq_len=args.max_seq_len,
        blend_alpha=args.blend_alpha,
        input_mode=args.input_mode,
        retention=args.retention,
        group_by='subject',
    )
    _, labels = dataset.get_groups_and_labels()
    if args.limit_records:
        keep = np.linspace(0, len(dataset) - 1, num=min(args.limit_records, len(dataset)), dtype=int)
        dataset.records = [dataset.records[i] for i in keep]
        _, labels = dataset.get_groups_and_labels()

    loader = make_loader(dataset, batch_size=args.batch_size, shuffle=False)
    ckpt_dir = Path(args.ckpt_dir)

    accumulated = np.zeros(len(dataset), dtype=np.float64)
    for fold in range(args.n_splits):
        model = load_checkpoint(ckpt_dir, args.ckpt_tag, fold, device)
        accumulated += predict_probabilities(model, loader, device)
        print(f'  checkpoint fold {fold} applied')
    probs = (accumulated / args.n_splits).astype(np.float32)

    auc, pr = auc_pr(labels, probs)
    return {'mode': 'ensemble', 'retention': args.retention, 'input_mode': args.input_mode,
            'n_records': len(dataset), 'auc': round(auc, 6), 'pr_auc': round(pr, 6)}


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f'[eval_ablation] mode={args.mode} device={device} ckpt_tag={args.ckpt_tag}')

    result = run_random_heatmap(args, device) if args.mode == 'random_heatmap' else run_ensemble(args, device)

    output_dir = Path(args.output_dir or args.ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f'eval_{args.mode}_{result["retention"]}_{result["input_mode"]}'
    write_rows(output_dir / f'{name}.csv', list(result), [result])

    print(f'\n[{name}] n={result["n_records"]} | AUC {result["auc"]:.4f} | PR-AUC {result["pr_auc"]:.4f}')


if __name__ == '__main__':
    main()
