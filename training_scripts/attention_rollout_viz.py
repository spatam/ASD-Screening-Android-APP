import torch
import numpy as np
import matplotlib.pyplot as plt
from asd_gaze.dataset import UnifiedGazeDataset
from asd_gaze.model import TwoStreamASD
import os

def attention_rollout(model, seq: torch.Tensor) -> np.ndarray:
    """
    Aggrega le attention matrix dei 4 layer Transformer.
    Ritorna importanza per token (T,).
    """
    attn_maps = []

    def hook(module, input, output):
        # output è (attn_output, attn_weights) se need_weights=True
        # TransformerEncoderLayer non espone weights direttamente
        # usiamo il forward manuale
        pass

    model.eval()
    with torch.no_grad():
        h = model.temporal_proj(seq)      # (1, T, 64)
        result = torch.eye(h.shape[1])    # identità (T, T)

        for layer in model.temporal_encoder.layers:
            # Accede direttamente all'attenzione del layer
            attn_out, attn_w = layer.self_attn(
                layer.norm1(h), layer.norm1(h), layer.norm1(h),
                need_weights=True, average_attn_weights=True
            )
            # attn_w: (1, T, T)
            A = attn_w.squeeze(0).cpu().numpy()   # (T, T)
            A = 0.5 * A + 0.5 * np.eye(len(A))   # residual connection
            A /= A.sum(axis=-1, keepdims=True)
            if isinstance(result, torch.Tensor):
                result = result.numpy()
            result = A @ result
            h = layer(h)

    if isinstance(result, torch.Tensor):
        result = result.numpy()
    importance = result.mean(axis=0)   # (T,)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance

def plot_rollout(model, ds, device, n_per_class=2, out_dir='runs/plots'):
    os.makedirs(out_dir, exist_ok=True)
    samples = {0: [], 1: []}
    for i in range(len(ds)):
        seq, hmap, label, group = ds[i]
        lbl = int(label.item())
        if len(samples[lbl]) < n_per_class:
            samples[lbl].append((seq.unsqueeze(0).to(device), group))
        if all(len(v) >= n_per_class for v in samples.values()):
            break

    for lbl, name in [(0, 'TD'), (1, 'ASD')]:
        for i, (seq, group) in enumerate(samples[lbl]):
            importance = attention_rollout(model, seq)  # (T,)

            fig, axes = plt.subplots(2, 1, figsize=(12, 5))
            fig.suptitle(f'Attention Rollout – {name} – {group}', fontsize=11)

            axes[0].plot(importance, color='#e74c3c' if lbl==1 else '#3498db',
                         linewidth=0.8, alpha=0.9)
            axes[0].fill_between(range(len(importance)), importance,
                                  alpha=0.25, color='#e74c3c' if lbl==1 else '#3498db')
            axes[0].set_ylabel('Importanza token'); axes[0].set_xlabel('Timestep')
            axes[0].set_title('Profilo di attenzione temporale')

            # Heatmap 1D
            axes[1].imshow(importance.reshape(1, -1), aspect='auto',
                           cmap='YlOrRd', vmin=0, vmax=1)
            axes[1].set_yticks([]); axes[1].set_xlabel('Timestep')
            axes[1].set_title('Heatmap temporale')

            plt.tight_layout()
            out_path = os.path.join(out_dir, f'rollout_{name.lower()}_{i}.png')
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
        plot_rollout(model, ds, device)
    else:
        print(f"Checkpoint {ckpt_path} non trovato. Aspetta la fine del fold 0.")
