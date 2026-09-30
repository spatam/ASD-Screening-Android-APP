"""One-off measurement of real MPS peak memory for the teacher (Phases 1-2,
shared architecture) and the student distillation step (Phase 3), on this
machine (Apple M4 Pro). Runs a few dozen real training batches through the
actual asd_gaze code paths -- enough for the caching allocator to reach its
steady-state high-water mark -- rather than a full multi-hour run, since the
architecture and batch size (not epoch count) determine the memory peak.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from asd_gaze.losses import binary_kd_loss, smooth_focal_loss
from asd_gaze.model import TwoStreamTeacher
from asd_gaze.student_model import StudentASD
from asd_gaze.train_common import build_protocol_dataset, fold_pos_weight, make_loader
from asd_gaze.utils import detect_device


def peak_gb(device: torch.device) -> tuple[float, float]:
    if device.type != 'mps':
        return 0.0, 0.0
    return (
        torch.mps.current_allocated_memory() / 1e9,
        torch.mps.driver_allocated_memory() / 1e9,
    )


def get_first_fold(args, device):
    dataset = build_protocol_dataset(args)
    groups, labels = dataset.get_groups_and_labels()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, _ = next(iter(splitter.split(np.zeros(len(dataset)), labels, groups)))
    loader = make_loader(dataset.with_indices(train_idx), batch_size=args.batch_size, shuffle=True)
    pos_weight = fold_pos_weight(labels, train_idx, device)
    return loader, pos_weight


def measure_teacher(args, device, n_batches: int) -> tuple[float, float, float]:
    loader, pos_weight = get_first_fold(args, device)
    model = TwoStreamTeacher(seq_feature_dim=3, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    peak_cur = peak_drv = 0.0
    t0 = time.time()
    n_done = 0
    for seq, visual, batch_labels, _ in loader:
        if n_done >= n_batches:
            break
        seq, visual, batch_labels = seq.to(device), visual.to(device), batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seq, visual)
        loss = smooth_focal_loss(logits, batch_labels, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.mps.synchronize()
        cur, drv = peak_gb(device)
        peak_cur, peak_drv = max(peak_cur, cur), max(peak_drv, drv)
        n_done += 1
    dt = time.time() - t0
    return peak_cur, peak_drv, dt / max(n_done, 1)


def measure_distill(args, device, n_batches: int, teacher_ckpt: str) -> tuple[float, float, float]:
    loader, pos_weight = get_first_fold(args, device)
    teacher = TwoStreamTeacher(seq_feature_dim=3, pretrained=True).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = StudentASD(seq_feature_dim=3, hidden=64, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-2)

    peak_cur = peak_drv = 0.0
    t0 = time.time()
    n_done = 0
    for seq, visual, batch_labels, _ in loader:
        if n_done >= n_batches:
            break
        seq, visual, batch_labels = seq.to(device), visual.to(device), batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(seq, visual)
        student_logits = student(seq, visual)
        loss = binary_kd_loss(
            student_logits=student_logits, teacher_logits=teacher_logits,
            hard_labels=batch_labels, temperature=4.0, alpha=0.7, pos_weight=pos_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        torch.mps.synchronize()
        cur, drv = peak_gb(device)
        peak_cur, peak_drv = max(peak_cur, cur), max(peak_drv, drv)
        n_done += 1
    dt = time.time() - t0
    return peak_cur, peak_drv, dt / max(n_done, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--huiyu-dir', required=True)
    p.add_argument('--max-seq-len', type=int, default=25)
    p.add_argument('--blend-alpha', type=float, default=0.55)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--protocol', default='subject')
    p.add_argument('--n-batches', type=int, default=30)
    p.add_argument('--teacher-ckpt', required=True, help='Real Phase-2 checkpoint to load as the frozen teacher for the distillation probe')
    args = p.parse_args()

    device = detect_device('mps')
    print(f'device={device}')

    cur, drv, spb = measure_teacher(args, device, args.n_batches)
    print(f'TEACHER (Phase 1/2 shared arch): peak current_allocated={cur:.3f} GB | peak driver_allocated={drv:.3f} GB | {spb*1000:.0f} ms/batch (bs={args.batch_size})')

    cur2, drv2, spb2 = measure_distill(args, device, args.n_batches, args.teacher_ckpt)
    print(f'STUDENT (Phase 3 distillation): peak current_allocated={cur2:.3f} GB | peak driver_allocated={drv2:.3f} GB | {spb2*1000:.0f} ms/batch (bs={args.batch_size})')


if __name__ == '__main__':
    main()
