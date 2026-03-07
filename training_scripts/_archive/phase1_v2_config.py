"""
Configurazione alternativa con regolarizzazione più forte,
da usare se phase1_v1 CV AUC < 0.82.
"""

CONFIG_V2 = {
    # Stessa architettura, ma:
    'lr': 1e-5,              # dimezzato rispetto a v1 (2e-5)
    'weight_decay': 5e-2,    # aumentato da 1e-2 (più L2)
    'dropout': 0.2,          # aumentato nel Transformer (era 0.1)
    'head_dropout': 0.4,     # aumentato nel fusion head (era 0.3)
    'epochs': 50,            # più epoche per compensare lr basso
    'warmup_epochs': 10,
    'tta_noise_std': 0.01,   # più augmentation durante TTA
    'label_smoothing': 0.1,  # smoother target: 0.9/0.1 invece di 1.0/0.0
    'run_tag': 'phase1_v2',
}

# Comando da lanciare:
CMD_V2 = """
python -m asd_gaze.train \\
    --data-dirs \\
        /home/mcasu/HD/max_project/cilia_2022_eye_tracking \\
        /home/mcasu/HD/max_project/huiyu_2019_eye_movements \\
        /home/mcasu/HD/max_project/qiao_he_2021_eyetracking \\
    --epochs 50 --batch-size 64 --lr 1e-5 \\
    --weight-decay 5e-2 \\
    --cv sgkf --n-splits 5 \\
    --warmup-epochs 10 --min-lr 1e-6 \\
    --spatial-backbone vit_b16 \\
    --tta-samples 5 --tta-noise-std 0.01 \\
    --label-smoothing 0.1 \\
    --output-dir runs --run-tag phase1_v2
"""
