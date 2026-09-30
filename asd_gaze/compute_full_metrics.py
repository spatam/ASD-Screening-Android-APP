from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .model import TwoStreamTeacher
from .student_model import StudentASD
from .train_common import add_protocol_argument, build_protocol_dataset, make_loader, write_rows
from .utils import aggregate_subject_probs, binary_classification_metrics, detect_device, youden_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute OOF metrics for all phases.')
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--phase1-tag', default='phase1_v1')
    parser.add_argument('--phase2-tag', default='phase2_v1')
    parser.add_argument('--phase3-tag', default='distill_v1')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--blend-alpha', type=float, default=0.55)
    parser.add_argument('--device', default='auto')
    add_protocol_argument(parser)
    return parser.parse_args()


def load_phase_model(phase: str, runs_dir: Path, tag: str, fold: int, device: torch.device):
    if phase in {'phase1', 'phase2'}:
        model = TwoStreamTeacher(seq_feature_dim=3, pretrained=False)
    else:
        model = StudentASD(seq_feature_dim=3, pretrained=False)
    ckpt = runs_dir / f'best_fold{fold}_{tag}.pth'
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_oof_probs(model_ctor, dataset, labels, groups, runs_dir, tag, n_splits, device, phase_name, per_subject: bool):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(dataset), dtype=np.float32)
    per_fold_rows: list[dict[str, float | int | str]] = []
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(dataset)), labels, groups)):
        model = model_ctor(runs_dir, tag, fold, device)
        loader = make_loader(dataset.with_indices(val_idx), batch_size=64, shuffle=False)
        batch_probs = []
        for seq, visual, _, _ in loader:
            seq = seq.to(device)
            visual = visual.to(device)
            probs = torch.sigmoid(model(seq, visual)).detach().cpu().numpy()
            batch_probs.append(probs)
        fold_probs = np.concatenate(batch_probs, axis=0)
        oof_probs[val_idx] = fold_probs

        fold_labels = labels[val_idx]
        fold_groups = groups[val_idx]
        row: dict[str, float | int | str] = {
            'phase': phase_name,
            'fold': fold,
            'records': int(len(val_idx)),
            'groups': int(len(np.unique(fold_groups))),
            'per_image_auc': round(float(roc_auc_score(fold_labels, fold_probs)), 6),
            'pr_auc': round(float(average_precision_score(fold_labels, fold_probs)), 6),
        }
        # aggregate_subject_probs assumes each group carries a single diagnostic
        # label, true when groups are subjects but false when groups are stimulus
        # images (held-out-stimulus protocol: every image has both ASD and TD
        # viewers). Under that protocol the aggregation is meaningless, so the
        # metric is left out. Averaging across mixed labels would hide the problem.
        row['per_subject_auc'] = (
            round(float(roc_auc_score(*aggregate_subject_probs(fold_groups, fold_labels, fold_probs))), 6)
            if per_subject else ''
        )
        per_fold_rows.append(row)
    return oof_probs, per_fold_rows


def summarise_phase(name: str, labels: np.ndarray, groups: np.ndarray, probs: np.ndarray, per_subject: bool) -> dict[str, float | str]:
    threshold = youden_threshold(labels, probs)
    per_image_auc = float(roc_auc_score(labels, probs))
    pr_auc = float(average_precision_score(labels, probs))
    img_metrics = binary_classification_metrics(labels, probs, threshold)
    per_subject_auc = ''
    if per_subject:
        subj_labels, subj_probs = aggregate_subject_probs(groups, labels, probs)
        per_subject_auc = round(float(roc_auc_score(subj_labels, subj_probs)), 6)
    return {
        'phase': name,
        'per_image_auc': round(per_image_auc, 6),
        'pr_auc': round(pr_auc, 6),
        'per_subject_auc': per_subject_auc,
        'threshold': round(float(threshold), 6),
        'accuracy': round(img_metrics['accuracy'], 6),
        'precision': round(img_metrics['precision'], 6),
        'recall': round(img_metrics['recall'], 6),
        'f1': round(img_metrics['f1'], 6),
    }


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    runs_dir = Path(args.runs_dir)
    dataset = build_protocol_dataset(args)
    groups, labels = dataset.get_groups_and_labels()

    phases = [
        ('phase1', args.phase1_tag),
        ('phase2', args.phase2_tag),
        ('phase3', args.phase3_tag),
    ]

    # Per-subject AUC only makes sense when groups are subjects: under the
    # held-out-stimulus protocol groups are images, which legitimately mix labels.
    per_subject = getattr(args, 'protocol', 'subject') in {'subject', 'subject_all'}

    summaries = []
    per_fold_rows: list[dict[str, float | int | str]] = []
    for phase_name, tag in phases:
        probs, phase_rows = collect_oof_probs(
            lambda runs_dir_, tag_, fold_, device_: load_phase_model(phase_name, runs_dir_, tag_, fold_, device_),
            dataset,
            labels,
            groups,
            runs_dir,
            tag,
            args.n_splits,
            device,
            phase_name,
            per_subject,
        )
        per_fold_rows.extend(phase_rows)
        summaries.append(summarise_phase(phase_name, labels, groups, probs, per_subject))

    write_rows(runs_dir / 'full_metrics_per_fold.csv', ['phase', 'fold', 'records', 'groups', 'per_image_auc', 'pr_auc', 'per_subject_auc'], per_fold_rows)
    write_rows(runs_dir / 'full_metrics_summary.csv', ['phase', 'per_image_auc', 'pr_auc', 'per_subject_auc', 'threshold', 'accuracy', 'precision', 'recall', 'f1'], summaries)


if __name__ == '__main__':
    main()
