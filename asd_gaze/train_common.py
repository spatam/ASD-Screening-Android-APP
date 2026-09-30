from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import HuiyuPerImageDataset


# Evaluation protocols. Each one sets both the subset of allowed files and
# the grouping key passed to StratifiedGroupKFold:
#
#   subject  : the paper's original protocol. Only files with exactly 14
#              subjects (2216 records), with folds grouped by subject-slot.
#              The same images appear in train and in validation.
#   stimulus : held-out-stimulus protocol requested by Reviewer 2. All files
#              (7296 records), with folds grouped by image identity, so train
#              and validation share no stimulus.
#   subject_all : deployment protocol. It matches real use of the app, which
#              always shows the same fixed set of 300 images to a new child.
#              The stimuli are therefore known in advance, and the model has
#              to generalise to new subjects. It uses all files (no exact-14
#              filter, so every image appears in both classes and the
#              image->label shortcut documented in Section III.B disappears)
#              with folds grouped by subject. The 14-slot cap avoids tail
#              groups of 1-18 records. Balancing per image brings
#              P(ASD|image) to exactly 0.5, so stimulus identity becomes as
#              informative as chance (AUC 0.500) and any result above it is,
#              by design, due to gaze.
PROTOCOL_SPECS: dict[str, dict] = {
    'subject': {'retention': 'retained', 'group_by': 'subject', 'max_slots': None,
                'balance_per_image': False},
    'stimulus': {'retention': 'all', 'group_by': 'image', 'max_slots': None,
                 'balance_per_image': False},
    'subject_all': {'retention': 'all', 'group_by': 'subject', 'max_slots': 14,
                    'balance_per_image': True},
}


def add_protocol_argument(parser) -> None:
    """Register ``--protocol`` with the same meaning in every phase."""
    parser.add_argument(
        '--protocol',
        default='subject',
        choices=sorted(PROTOCOL_SPECS),
        help="subject: original protocol (2216 records, grouped by subject). "
             "stimulus: held-out-stimulus (7296 records, grouped by image). "
             "subject_all: deployment protocol (all files, 14-slot cap, "
             "grouped by subject).",
    )


def build_protocol_dataset(args, input_mode: str = 'blend') -> HuiyuPerImageDataset:
    """Build the dataset for the chosen protocol.

    The ``subject`` default reproduces the previous behaviour exactly, so
    phases that already ran stay bit-identical without the flag.
    """
    spec = PROTOCOL_SPECS[getattr(args, 'protocol', 'subject')]
    return HuiyuPerImageDataset(
        args.huiyu_dir,
        max_seq_len=args.max_seq_len,
        blend_alpha=args.blend_alpha,
        input_mode=input_mode,
        retention=spec['retention'],
        group_by=spec['group_by'],
        max_slots=spec.get('max_slots'),
        balance_per_image=spec.get('balance_per_image', False),
    )


def collate_fn(batch):
    seqs, visuals, labels, groups = zip(*batch)
    return torch.stack(seqs), torch.stack(visuals), torch.stack(labels), list(groups)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    if num_workers <= 0:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, max(1, cpu_count // 2))

    # If the last training batch would hold a single sample, the student's
    # BatchNorm layers raise "Expected more than 1 value per channel", since
    # batch variance is undefined for one element. We drop that leftover batch
    # and only that one. The condition covers just the failing case (remainder
    # exactly 1, and only in training), so the configurations already
    # reproduced, where the remainder is not 1, keep their training dynamics.
    drop_last = bool(shuffle) and len(dataset) % batch_size == 1

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': False,
        'collate_fn': collate_fn,
        'drop_last': drop_last,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    return DataLoader(
        **loader_kwargs,
    )


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        min_ratio = min_lr / base_lr
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def fold_pos_weight(labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    train_labels = labels[train_indices]
    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)


@torch.no_grad()
def predict_probabilities(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []
    for seq, visual, _, _ in loader:
        seq = seq.to(device)
        visual = visual.to(device)
        probs = torch.sigmoid(model(seq, visual)).detach().cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def write_rows(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def auc_pr(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(labels, probs)), float(average_precision_score(labels, probs))
