import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_cv(csv_path: str, run_tag: str, out_dir: str = 'runs/plots'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    folds  = df['fold'].values
    aucs   = df['best_auc'].values
    mean_a = aucs.mean()
    std_a  = aucs.std()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Fold-wise AUC ---
    ax = axes[0]
    colors = ['#e74c3c' if a < 0.75 else '#2ecc71' for a in aucs]
    ax.bar(folds, aucs, color=colors, alpha=0.85, width=0.6)
    ax.axhline(mean_a, color='#2c3e50', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_a:.4f} ± {std_a:.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Best Validation ROC-AUC')
    ax.set_title('Fold-wise ROC-AUC – Phase 1')
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(folds)
    ax.legend(fontsize=9)

    # --- Distribution ---
    ax = axes[1]
    ax.hist(aucs, bins=5, color='#3498db', alpha=0.8, edgecolor='white')
    ax.axvline(mean_a, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_a:.4f}')
    ax.set_xlabel('ROC-AUC')
    ax.set_ylabel('Count')
    ax.set_title('AUC Distribution – Phase 1')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'cv_results_{run_tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    csv_path = 'runs/cv_results_phase1_v1.csv'
    if os.path.exists(csv_path):
        plot_cv(csv_path, run_tag='phase1_v1')
    else:
        print(f"File {csv_path} not found. Wait for training to finish.")
