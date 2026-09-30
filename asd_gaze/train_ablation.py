"""Training for the ablations reported in the paper (Sections VI.A and VI.B).

Each ablation combines three independent axes, which ``HuiyuPerImageDataset``
and ``SpatialOnlyTeacher`` already support:

    ablation   -> which information enters the model
    protocol   -> how to build the folds
    retention  -> which scanpath files to allow (derived from the protocol)

Combinations covered by the paper:

    ablation=spatial_only protocol=subject   -> CV 0.9855+-0.0026, OOF 0.9852
    ablation=image_only   protocol=subject   -> CV 0.9827+-0.0007, OOF 0.9828
    ablation=spatial_only protocol=stimulus  -> CV 0.5471+-0.0091, OOF 0.5464
    ablation=image_only   protocol=stimulus  -> CV 0.5045+-0.0050, OOF 0.5032

The random-heatmap control does not appear here because by design it involves
no training. It reuses the spatial-only checkpoints, and ``eval_ablation``
implements it.

Typical use:

    python -m asd_gaze.train_ablation --huiyu-dir <dir> \
        --ablation spatial_only --protocol subject --output-dir runs_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .ablation_models import SpatialOnlyTeacher
from .dataset import HuiyuPerImageDataset
from .model import TwoStreamTeacher
from .train_common import (
    PROTOCOL_SPECS,
    auc_pr,
    cosine_scheduler,
    fold_pos_weight,
    make_loader,
    predict_probabilities,
    write_rows,
)
from .utils import aggregate_subject_probs, binary_classification_metrics, count_parameters, detect_device, youden_threshold


# Dataset input_mode and model class for each ablation.
ABLATION_SPECS: dict[str, dict[str, object]] = {
    # Full teacher, used only as a reference/control. It matches Phase 1.
    'full': {'input_mode': 'blend', 'model': TwoStreamTeacher, 'spatial_only': False},
    # Spatial branch only, on the blended stimulus + real heatmap image.
    'spatial_only': {'input_mode': 'blend', 'model': SpatialOnlyTeacher, 'spatial_only': True},
    # Stimulus only, with no gaze information.
    'image_only': {'input_mode': 'image_only', 'model': SpatialOnlyTeacher, 'spatial_only': True},
}

# train_common defines PROTOCOL_SPECS and shares it with phases 1-3, so
# "stimulus" means exactly the same thing everywhere.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the ablations from the paper.')
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--ablation', required=True, choices=sorted(ABLATION_SPECS))
    parser.add_argument('--protocol', default='subject', choices=sorted(PROTOCOL_SPECS))
    parser.add_argument('--output-dir', default='runs_ablation')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-temporal', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--blend-alpha', type=float, default=0.55)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--temporal-ckpt', default='',
                        help='Checkpoint from pretrain_temporal that initialises the temporal branch '
                             '(cross-dataset pre-training ablation). Spatial-only variants ignore it.')
    parser.add_argument('--pretrained', type=int, default=1, help='0 for a backbone without pre-trained weights (smoke test only)')
    parser.add_argument('--limit-records', type=int, default=0, help='>0 to truncate the dataset (smoke test only)')
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, spatial_only: bool, args: argparse.Namespace) -> torch.optim.Optimizer:
    """AdamW over only the parameters that receive a gradient.

    The spatial-only variants bypass the temporal branch, so the graph never
    reaches its weights. Adding them to the optimizer would leave them
    unchanged. It would still make the count of optimised parameters misleading.
    """
    fusion_params = (
        list(model.fusion_in.parameters())
        + list(model.fusion_norm.parameters())
        + list(model.fusion_out.parameters())
    )
    param_groups: list[dict] = [{'params': fusion_params, 'lr': args.lr}]
    if not spatial_only:
        param_groups.insert(
            0,
            {
                'params': list(model.temporal_proj.parameters()) + list(model.temporal_encoder.parameters()),
                'lr': args.lr_temporal,
            },
        )
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    spec = ABLATION_SPECS[args.ablation]
    protocol = PROTOCOL_SPECS[args.protocol]
    run_tag = args.run_tag or f'{args.ablation}_{args.protocol}'

    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = HuiyuPerImageDataset(
        args.huiyu_dir,
        max_seq_len=args.max_seq_len,
        blend_alpha=args.blend_alpha,
        input_mode=str(spec['input_mode']),
        retention=protocol['retention'],
        group_by=protocol['group_by'],
    )
    groups, labels = dataset.get_groups_and_labels()

    if args.limit_records:
        keep = np.linspace(0, len(dataset) - 1, num=min(args.limit_records, len(dataset)), dtype=int)
        dataset.records = [dataset.records[i] for i in keep]
        groups, labels = dataset.get_groups_and_labels()
        print(f'[ablation] SMOKE: dataset truncated to {len(dataset)} records')

    print(f'[ablation] {run_tag} | device={device} | records={len(dataset)} | groups={len(set(groups))}')

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_rows: list[dict] = []
    oof_probs = np.zeros(len(dataset), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(dataset)), labels, groups)):
        print(f'\n=== {run_tag} | fold {fold} ===')
        train_loader = make_loader(dataset.with_indices(train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = make_loader(dataset.with_indices(val_idx), batch_size=args.batch_size, shuffle=False)

        model_cls = spec['model']
        model = model_cls(seq_feature_dim=3, pretrained=bool(args.pretrained)).to(device)  # type: ignore[operator]

        if args.temporal_ckpt:
            if spec['spatial_only']:
                # The temporal branch takes no part in the spatial-only forward pass.
                # Loading its weights would change nothing, so we flag the inconsistency.
                print('[ablation] --temporal-ckpt ignored because the variant is spatial-only.')
            else:
                payload = torch.load(args.temporal_ckpt, map_location=device)
                model.temporal_proj.load_state_dict(payload['temporal_proj'])
                model.temporal_encoder.load_state_dict(payload['temporal_encoder'])
                print(f'[ablation] temporal branch initialised from {args.temporal_ckpt}')
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=fold_pos_weight(labels, train_idx, device))
        optimizer = build_optimizer(model, bool(spec['spatial_only']), args)
        scheduler = cosine_scheduler(optimizer, args.warmup_epochs, args.epochs, args.lr, min_lr=1e-6)

        best_auc = -1.0
        best_epoch = -1
        best_path = output_dir / f'best_fold{fold}_{run_tag}.pth'
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for seq, visual, batch_labels, _ in train_loader:
                seq = seq.to(device)
                visual = visual.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(seq, visual)
                loss = criterion(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_labels)

            scheduler.step()
            val_probs = predict_probabilities(model, val_loader, device)
            val_auc, val_pr = auc_pr(labels[val_idx], val_probs)
            train_loss = total_loss / max(len(train_idx), 1)
            print(f'ep {epoch:02d} | train_loss {train_loss:.4f} | val_auc {val_auc:.4f} | val_pr {val_pr:.4f}')

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
                if patience >= args.patience:
                    print(f'early stop @ epoch {epoch}')
                    break

        # Recompute the out-of-fold predictions with the best checkpoint of the
        # fold, to match how the paper reports OOF metrics.
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        oof_probs[val_idx] = predict_probabilities(model, val_loader, device)

        fold_rows.append(
            {
                'fold': fold,
                'best_auc': round(best_auc, 6),
                'best_epoch': best_epoch,
                'n_train': int(len(train_idx)),
                'n_val': int(len(val_idx)),
                'trainable_params': count_parameters(model, trainable_only=True),
                'total_params': count_parameters(model),
            }
        )

    write_rows(
        output_dir / f'cv_results_{run_tag}.csv',
        ['fold', 'best_auc', 'best_epoch', 'n_train', 'n_val', 'trainable_params', 'total_params'],
        fold_rows,
    )

    # OOF summary over the concatenated predictions, as in Table "sanity checks".
    oof_auc, oof_pr = auc_pr(labels, oof_probs)
    threshold = youden_threshold(labels, oof_probs)
    metrics = binary_classification_metrics(labels, oof_probs, threshold)
    fold_aucs = [row['best_auc'] for row in fold_rows]

    summary = {
        'ablation': args.ablation,
        'protocol': args.protocol,
        'n_records': len(dataset),
        'cv_auc_mean': round(float(np.mean(fold_aucs)), 6),
        'cv_auc_std': round(float(np.std(fold_aucs)), 6),
        'oof_auc': round(oof_auc, 6),
        'oof_pr_auc': round(oof_pr, 6),
        'accuracy': round(metrics['accuracy'], 6),
        'f1': round(metrics['f1'], 6),
    }
    # Per-subject AUC only makes sense when the groups are subjects.
    if args.protocol == 'subject':
        subj_labels, subj_probs = aggregate_subject_probs(groups, labels, oof_probs)
        summary['per_subject_auc'] = round(float(auc_pr(subj_labels, subj_probs)[0]), 6)
    else:
        summary['per_subject_auc'] = ''

    write_rows(output_dir / f'summary_{run_tag}.csv', list(summary), [summary])
    np.save(output_dir / f'oof_probs_{run_tag}.npy', oof_probs)

    print(f'\n[{run_tag}] CV AUC {summary["cv_auc_mean"]:.4f}+-{summary["cv_auc_std"]:.4f} | '
          f'OOF AUC {summary["oof_auc"]:.4f} | PR-AUC {summary["oof_pr_auc"]:.4f}')


if __name__ == '__main__':
    main()
