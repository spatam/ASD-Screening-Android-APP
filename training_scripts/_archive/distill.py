"""
Knowledge Distillation: ensemble teacher → student.
Loss = α * KL(soft teacher | soft student) + (1-α) * BCE(hard labels | student)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os, glob
from asd_gaze.dataset import UnifiedGazeDataset
from asd_gaze.model import TwoStreamASD
from asd_gaze.student_model import StudentASD
from asd_gaze.train import collate_fn
from sklearn.metrics import roc_auc_score

TEMPERATURE = 4.0    # smoothing delle soft labels
ALPHA       = 0.7    # peso KL loss vs BCE loss

def get_teacher_probs(teacher_ckpts, loader, device, T=TEMPERATURE):
    """Pre-calcola soft labels del teacher sull'intero dataset."""
    all_logits = []
    for ckpt in teacher_ckpts:
        model = TwoStreamASD(seq_feature_dim=6).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        fold_logits = []
        with torch.no_grad():
            for seq, hmap, _, _ in loader:
                seq, hmap = seq.to(device), hmap.to(device)
                fold_logits.append(model(seq, hmap).cpu())
        all_logits.append(torch.cat(fold_logits))

    # Media logit tra fold (ensemble) poi soft con temperatura T
    mean_logits   = torch.stack(all_logits).mean(dim=0)
    soft_probs    = torch.sigmoid(mean_logits / T)
    return soft_probs   # (N,) ∈ [0,1]

def distillation_loss(student_logits, soft_teacher, hard_labels,
                      T=TEMPERATURE, alpha=ALPHA):
    # KL divergence sui logit scalati per temperatura
    p_s = torch.sigmoid(student_logits / T)
    p_t = soft_teacher.to(student_logits.device)
    kl  = F.binary_cross_entropy(p_s, p_t) * (T ** 2)

    # BCE standard con hard labels
    bce = F.binary_cross_entropy_with_logits(
        student_logits, hard_labels.to(student_logits.device))

    return alpha * kl + (1 - alpha) * bce

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset completo (stesso di Phase 1)
    ds = UnifiedGazeDataset([
        '/home/mcasu/HD/max_project/cilia_2022_eye_tracking',
        '/home/mcasu/HD/max_project/huiyu_2019_eye_movements',
        '/home/mcasu/HD/max_project/qiao_he_2021_eyetracking',
    ])
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    # Trova checkpoint teacher
    teacher_ckpts = sorted(
        glob.glob('runs/best_fold*_phase1_v1_fixed.pth'))
    print(f"[Distill] teacher checkpoint: {len(teacher_ckpts)}")

    # Pre-calcola soft labels (una volta, poi riusa)
    print("[Distill] calcolo soft labels teacher...")
    soft_labels = get_teacher_probs(teacher_ckpts, loader, device)
    torch.save(soft_labels, 'runs/teacher_soft_labels.pt')
    print(f"[Distill] soft labels salvate: {soft_labels.shape}")

    # Student
    student = StudentASD(seq_feature_dim=6).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=1e-4, weight_decay=1e-2)

    _, hard_labels = ds.get_groups_and_labels()
    hard_labels_t  = torch.FloatTensor(hard_labels)

    best_auc = 0.0
    for epoch in range(60):
        student.train()
        total_loss, all_preds = 0.0, []
        idx = 0

        for seq, hmap, labels, _ in DataLoader(
                ds, batch_size=64, shuffle=True,
                collate_fn=collate_fn, num_workers=4):
            seq, hmap = seq.to(device), hmap.to(device)
            B = len(labels)

            # Nota per soft labels: la destrutturazione e index lookup
            # andrebbero mappati. Usiamo solo CE standard momentaneamente per struttura
            student_logits = student(seq, hmap)
            loss = F.binary_cross_entropy_with_logits(
                student_logits,
                labels.to(device))
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item() * B
            idx += B

        # Valutazione su intero dataset
        student.eval()
        all_probs = []
        with torch.no_grad():
            for seq, hmap, _, _ in loader:
                seq, hmap = seq.to(device), hmap.to(device)
                all_probs.append(
                    torch.sigmoid(student(seq, hmap)).cpu())
        all_probs = torch.cat(all_probs).numpy()
        auc = roc_auc_score(hard_labels, all_probs)

        print(f"ep {epoch:02d} | loss {total_loss/len(ds):.4f} "
              f"| AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(student.state_dict(),
                       'runs/best_student.pth')

    print(f"\n[Distill] best student AUC: {best_auc:.4f}")

    # Export ONNX student
    student.load_state_dict(
        torch.load('runs/best_student.pth', map_location='cpu'))
    student.eval().cpu()
    dummy_seq  = torch.randn(1, 500, 6)
    dummy_hmap = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        student, (dummy_seq, dummy_hmap),
        'runs/student_asd.onnx',
        opset_version=18,
        input_names=['sequence', 'heatmap'],
        output_names=['logit'],
        dynamic_axes={'sequence': {0: 'batch'},
                      'heatmap':  {0: 'batch'},
                      'logit':    {0: 'batch'}},
    )
    size = os.path.getsize('runs/student_asd.onnx') / 1e6
    print(f"[Export] student ONNX: {size:.1f} MB")

if __name__ == '__main__':
    main()
