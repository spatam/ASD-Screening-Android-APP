from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .losses import binary_kd_loss
from .model import TwoStreamTeacher
from .student_model import StudentASD
from .train_common import add_protocol_argument, auc_pr, build_protocol_dataset, cosine_scheduler, fold_pos_weight, make_loader, predict_probabilities, write_rows
from .utils import count_parameters, detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 3: student distillation.')
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--teacher-ckpt-pattern', required=True)
    parser.add_argument('--output-dir', default='runs')
    parser.add_argument('--run-tag', default='distill_v1')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--blend-alpha', type=float, default=0.55)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--student-hidden', type=int, default=64)
    parser.add_argument('--device', default='auto')
    add_protocol_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_protocol_dataset(args)
    groups, labels = dataset.get_groups_and_labels()
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_rows: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(dataset)), labels, groups)):
        print(f'\n=== Phase 3 | fold {fold} ===')
        train_loader = make_loader(dataset.with_indices(train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = make_loader(dataset.with_indices(val_idx), batch_size=args.batch_size, shuffle=False)

        teacher = TwoStreamTeacher(seq_feature_dim=3, pretrained=True).to(device)
        teacher_ckpt = Path(args.teacher_ckpt_pattern.format(fold=fold))
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f'Teacher checkpoint not found: {teacher_ckpt}')
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

        student = StudentASD(seq_feature_dim=3, hidden=args.student_hidden, pretrained=True).to(device)
        pos_weight = fold_pos_weight(labels, train_idx, device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = cosine_scheduler(optimizer, args.warmup_epochs, args.epochs, args.lr, min_lr=1e-6)

        best_auc = -1.0
        best_epoch = -1
        best_path = output_dir / f'best_fold{fold}_{args.run_tag}.pth'
        patience = 0

        for epoch in range(args.epochs):
            student.train()
            total_loss = 0.0
            for seq, visual, batch_labels, _ in train_loader:
                seq = seq.to(device)
                visual = visual.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_logits = teacher(seq, visual)
                student_logits = student(seq, visual)
                loss = binary_kd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    hard_labels=batch_labels,
                    temperature=args.temperature,
                    alpha=args.alpha,
                    pos_weight=pos_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_labels)

            scheduler.step()
            val_probs = predict_probabilities(student, val_loader, device)
            val_auc, val_pr = auc_pr(labels[val_idx], val_probs)
            train_loss = total_loss / max(len(train_idx), 1)
            print(f'ep {epoch:02d} | train_loss {train_loss:.4f} | val_auc {val_auc:.4f} | val_pr {val_pr:.4f}')

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                patience = 0
                torch.save(student.state_dict(), best_path)
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
                'trainable_params': count_parameters(student, trainable_only=True),
                'total_params': count_parameters(student),
            }
        )

    write_rows(output_dir / f'cv_results_{args.run_tag}.csv', ['fold', 'best_auc', 'best_epoch', 'trainable_params', 'total_params'], fold_rows)


if __name__ == '__main__':
    main()
