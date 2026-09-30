from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .losses import smooth_focal_loss
from .model import TwoStreamTeacher
from .train_common import add_protocol_argument, auc_pr, build_protocol_dataset, cosine_scheduler, fold_pos_weight, make_loader, predict_probabilities, write_rows
from .utils import count_parameters, detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 2: augmentation-regularised teacher.')
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--temporal-ckpt', required=True, help='Pattern such as runs/best_fold{fold}_phase1_v1.pth')
    parser.add_argument('--output-dir', default='runs')
    parser.add_argument('--run-tag', default='phase2_v1')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-temporal', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--blend-alpha', type=float, default=0.55)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--gaze-jitter', type=float, default=0.02)
    parser.add_argument('--fix-dropout', type=float, default=0.1)
    parser.add_argument('--device', default='auto')
    add_protocol_argument(parser)
    return parser.parse_args()


def sequence_mixup(seq: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or len(seq) < 2:
        return seq, labels
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(len(seq), device=seq.device)
    mixed_seq = lam * seq + (1.0 - lam) * seq[perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_seq, mixed_labels


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_protocol_dataset(args)
    groups, labels = dataset.get_groups_and_labels()
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_rows: list[dict] = []

    augment = {'gaze_jitter': args.gaze_jitter, 'fix_dropout': args.fix_dropout}

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(dataset)), labels, groups)):
        print(f'\n=== Phase 2 | fold {fold} ===')
        train_loader = make_loader(dataset.with_indices(train_idx, augment=augment), batch_size=args.batch_size, shuffle=True)
        val_loader = make_loader(dataset.with_indices(val_idx), batch_size=args.batch_size, shuffle=False)

        model = TwoStreamTeacher(seq_feature_dim=3, pretrained=True).to(device)
        phase1_ckpt = Path(args.temporal_ckpt.format(fold=fold))
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f'Phase 1 checkpoint not found: {phase1_ckpt}')
        model.load_state_dict(torch.load(phase1_ckpt, map_location=device))

        pos_weight = fold_pos_weight(labels, train_idx, device)
        optimizer = torch.optim.AdamW(
            [
                {'params': list(model.temporal_proj.parameters()) + list(model.temporal_encoder.parameters()), 'lr': args.lr_temporal},
                {'params': list(model.fusion_in.parameters()) + list(model.fusion_norm.parameters()) + list(model.fusion_out.parameters()), 'lr': args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = cosine_scheduler(optimizer, args.warmup_epochs, args.epochs, args.lr, min_lr=1e-6)

        best_auc = -1.0
        best_epoch = -1
        best_path = output_dir / f'best_fold{fold}_{args.run_tag}.pth'
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for seq, visual, batch_labels, _ in train_loader:
                seq = seq.to(device)
                visual = visual.to(device)
                batch_labels = batch_labels.to(device)

                seq, mixed_labels = sequence_mixup(seq, batch_labels, args.mixup_alpha)

                optimizer.zero_grad(set_to_none=True)
                logits = model(seq, visual)
                loss = smooth_focal_loss(
                    logits,
                    mixed_labels,
                    epsilon=args.label_smoothing,
                    gamma=args.focal_gamma,
                    pos_weight=pos_weight,
                )
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

        fold_rows.append(
            {
                'fold': fold,
                'best_auc': round(best_auc, 6),
                'best_epoch': best_epoch,
                'trainable_params': count_parameters(model, trainable_only=True),
                'total_params': count_parameters(model),
            }
        )

    write_rows(output_dir / f'cv_results_{args.run_tag}.csv', ['fold', 'best_auc', 'best_epoch', 'trainable_params', 'total_params'], fold_rows)


if __name__ == '__main__':
    main()
