import torch, os, glob, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from asd_gaze.dataset import UnifiedGazeDataset
from asd_gaze.model import TwoStreamASD
from asd_gaze.train import collate_fn

def ensemble_predict(ckpt_paths, loader, device, tta_samples=5, tta_std=0.005):
    all_probs = []
    for ckpt in ckpt_paths:
        model = TwoStreamASD(seq_feature_dim=6).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for seq, hmap, labels, _ in loader:
                seq, hmap = seq.to(device), hmap.to(device)
                probs = torch.zeros(len(labels), device=device)
                for _ in range(tta_samples):
                    noise = torch.randn_like(seq) * tta_std
                    probs += torch.sigmoid(model(seq + noise, hmap))
                probs /= tta_samples
                fold_probs.extend(probs.cpu().numpy())
        all_probs.append(fold_probs)
    return np.mean(all_probs, axis=0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = UnifiedGazeDataset([
        '/home/mcasu/HD/max_project/cilia_2022_eye_tracking',
        '/home/mcasu/HD/max_project/huiyu_2019_eye_movements',
        '/home/mcasu/HD/max_project/qiao_he_2021_eyetracking',
    ])
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)
    _, labels = ds.get_groups_and_labels()

    ckpts = sorted(glob.glob('runs/best_fold*_phase1_v1.pth'))
    print(f"Checkpoint trovati: {len(ckpts)}")

    if len(ckpts) > 0:
        probs = ensemble_predict(ckpts, loader, device)
        auc  = roc_auc_score(labels, probs)
        prauc = average_precision_score(labels, probs)
        print(f"Ensemble ROC-AUC:  {auc:.4f}")
        print(f"Ensemble PR-AUC:   {prauc:.4f}")
    else:
        print("Nessun checkpoint trovato.")
