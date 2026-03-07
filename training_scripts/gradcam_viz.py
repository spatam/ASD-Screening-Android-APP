"""
Genera figure Grad-CAM reali per 4 campioni (2 ASD, 2 TD).
Output: runs/plots/gradcam_asd_0.png, gradcam_td_0.png ...
Richiede il modello Phase 1 (usa fold 0 come default).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from asd_gaze.dataset import UnifiedGazeDataset
from asd_gaze.model import TwoStreamASD
from asd_gaze.train import collate_fn
import os

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook sull'ultimo blocco ViT (block[-1].norm1)
        target = model.spatial_backbone.blocks[-1].norm1
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()   # (B, N_tokens, 768)

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, seq, hmap, class_idx=1):
        self.model.zero_grad()
        logit = self.model(seq, hmap)
        logit.backward(torch.ones_like(logit) * class_idx)

        # Pool gradients sui token (esclude CLS)
        grads  = self.gradients[:, 1:, :]        # (B, 196, 768)
        acts   = self.activations[:, 1:, :]
        weights = grads.mean(dim=1, keepdim=True) # (B, 1, 768)
        cam = (weights * acts).sum(dim=2)         # (B, 196)
        cam = torch.relu(cam)

        B = cam.shape[0]
        H = W = int(cam.shape[1] ** 0.5)         # 14x14
        cam = cam.reshape(B, H, W).cpu().numpy()

        # Resize a 224x224
        import cv2
        cams_resized = []
        for b in range(B):
            c = cam[b]
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            c = cv2.resize(c, (224, 224))
            cams_resized.append(c)
        return np.stack(cams_resized)

def visualize_gradcam(model, ds, device, n_per_class=2, out_dir='runs/plots'):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    gcam = GradCAM(model)

    # Prendi n campioni per classe
    samples = {0: [], 1: []}
    for i in range(len(ds)):
        seq, hmap, label, group = ds[i]
        lbl = int(label.item())
        if len(samples[lbl]) < n_per_class:
            samples[lbl].append((seq.unsqueeze(0), hmap.unsqueeze(0), group))
        if all(len(v) >= n_per_class for v in samples.values()):
            break

    for lbl, label_name in [(0, 'TD'), (1, 'ASD')]:
        for i, (seq, hmap, group) in enumerate(samples[lbl]):
            seq  = seq.to(device)
            hmap = hmap.to(device).requires_grad_(True)

            cam = gcam.compute(seq, hmap)[0]           # (224, 224)
            hmap_np = hmap[0].mean(0).cpu().detach().numpy()  # media 3 canali

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f'{label_name} – {group}', fontsize=12)

            axes[0].imshow(hmap_np, cmap='hot')
            axes[0].set_title('Gaze Heatmap'); axes[0].axis('off')

            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('Grad-CAM'); axes[1].axis('off')

            overlay = 0.5 * hmap_np / (hmap_np.max() + 1e-8) + 0.5 * cam
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
            axes[2].imshow(overlay, cmap='jet')
            axes[2].set_title('Overlay'); axes[2].axis('off')

            plt.tight_layout()
            out_path = os.path.join(out_dir, f'gradcam_{label_name.lower()}_{i}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Salvato: {out_path}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = UnifiedGazeDataset([
        '/home/mcasu/HD/max_project/cilia_2022_eye_tracking',
        '/home/mcasu/HD/max_project/huiyu_2019_eye_movements',
        '/home/mcasu/HD/max_project/qiao_he_2021_eyetracking',
    ])
    model = TwoStreamASD(seq_feature_dim=6).to(device)
    ckpt_path = 'runs/best_fold0_phase1_v1.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        visualize_gradcam(model, ds, device)
    else:
        print(f"Checkpoint {ckpt_path} non trovato. Aspetta la fine del fold 0.")
