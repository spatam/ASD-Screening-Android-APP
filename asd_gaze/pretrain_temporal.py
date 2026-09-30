"""Cross-dataset pre-training of the temporal encoder (ablation row 1-dagger).

In the paper, pre-training on external gaze datasets can initialise the
teacher's temporal branch. The ablation shows that the difference from random
initialisation is negligible (|Delta AUC| < 0.001).

Source
------
He et al., 2021 (``--he-dir``)
    Three MATLAB v5 files (HFA/LFA/TD). The fixations are in
    ``sub(i).trial_proData(t).fixation``, with columns (x, y, duration_us, index).
    Labels come from the ``sub(i).function`` field: 1=LFA, 2=HFA -> ASD; 3=TD -> control.
    The files hold 74 subjects (26 HFA + 24 LFA + 24 TD) and 58,886 fixations.
    These are the dataset's actual counts. They replace the 66 subjects and
    58,439 fixations that the first draft of the paper reported by mistake.

We leave out Cilia et al., 2022 on purpose. Its public release has recordings
only for participants 1-25, and ``Metadata_Participants.csv`` lists all of them
as ASD. The TD participants (IDs 30-59) are missing. With a single class,
supervised pre-training is undefined.

We normalise the sequences exactly as ``HuiyuPerImageDataset`` does (local
min-max on x/y, log-normalisation of the duration in ms). Otherwise the
pre-trained weights would not transfer to the teacher.

Typical use:

    python -m asd_gaze.pretrain_temporal \
        --he-dir /path/to/final_data --out runs_ablation/temporal_pretrained.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .utils import detect_device, duration_log_normalize, local_minmax, pad_or_truncate


HE_FILES = {'HFA_Group.mat': 1, 'LFA_Group.mat': 1, 'TD_Group.mat': 0}
MICROSECONDS_PER_MS = 1000.0


def load_he_sequences(he_dir: Path, max_seq_len: int, min_fixations: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (sequences, labels, subject groups) from the three He 2021 MATLAB files."""
    import scipy.io as sio

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for fname, label in HE_FILES.items():
        path = he_dir / fname
        if not path.exists():
            raise FileNotFoundError(f'Missing He 2021 file: {path}')

        payload = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
        subjects = np.atleast_1d(payload['sub'])
        for subject_index, subject in enumerate(subjects):
            subject_key = f'he_{path.stem}_{subject_index:03d}'
            for trial in np.atleast_1d(subject.trial_proData):
                fixations = np.atleast_2d(np.asarray(trial.fixation, dtype=np.float64))
                if fixations.size == 0 or fixations.shape[1] < 3 or len(fixations) < min_fixations:
                    continue

                seq = np.zeros((len(fixations), 3), dtype=np.float32)
                seq[:, 0] = local_minmax(fixations[:, 0])
                seq[:, 1] = local_minmax(fixations[:, 1])
                # He durations are in microseconds. The teacher expects ms.
                seq[:, 2] = duration_log_normalize(fixations[:, 2] / MICROSECONDS_PER_MS)

                sequences.append(pad_or_truncate(seq, max_seq_len))
                labels.append(label)
                groups.append(subject_key)

        del payload

    if not sequences:
        raise RuntimeError(f'No sequences extracted from {he_dir}.')

    return np.stack(sequences), np.asarray(labels, dtype=np.float32), np.asarray(groups)


class TemporalPretrainer(nn.Module):
    """The teacher's temporal encoder with a throwaway linear head.

    ``temporal_proj`` and ``temporal_encoder`` have the same names and shapes as
    in ``TwoStreamTeacher``, so the teacher can load the checkpoint directly.
    """

    def __init__(self, seq_feature_dim: int = 3, temporal_dim: int = 64,
                 temporal_layers: int = 3, temporal_heads: int = 4, temporal_ff_dim: int = 128) -> None:
        super().__init__()
        self.temporal_proj = nn.Linear(seq_feature_dim, temporal_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=temporal_heads,
            dim_feedforward=temporal_ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.head = nn.Linear(temporal_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_proj(sequence)
        padding_mask = sequence.abs().sum(dim=-1) == 0
        # An all-zero sequence would mask every token and produce NaN in the
        # attention softmax, so we always keep at least the first token unmasked.
        padding_mask[:, 0] = False
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).float().unsqueeze(-1)
        pooled = (temporal * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset pre-training of the temporal encoder.")
    parser.add_argument('--he-dir', required=True, help='Folder with HFA_Group.mat / LFA_Group.mat / TD_Group.mat')
    parser.add_argument('--out', required=True, help='Path of the checkpoint to write')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-seq-len', type=int, default=25)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit-records', type=int, default=0, help='>0 to truncate (smoke test only)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)

    sequences, labels, groups = load_he_sequences(Path(args.he_dir).expanduser().resolve(), args.max_seq_len)
    if args.limit_records:
        keep = np.linspace(0, len(sequences) - 1, num=min(args.limit_records, len(sequences)), dtype=int)
        sequences, labels, groups = sequences[keep], labels[keep], groups[keep]
    print(f'[pretrain] sequences {len(sequences)} | ASD {int(labels.sum())} | TD {int((labels == 0).sum())} '
          f'| subjects {len(set(groups))}')

    # Split by subject, so no subject appears in both train and validation.
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.default_rng(42)
    rng.shuffle(unique_groups)
    n_val = max(1, int(len(unique_groups) * args.val_fraction))
    val_groups = set(unique_groups[:n_val])
    is_val = np.array([group in val_groups for group in groups])

    x_train = torch.from_numpy(sequences[~is_val])
    y_train = torch.from_numpy(labels[~is_val])
    x_val = torch.from_numpy(sequences[is_val]).to(device)
    y_val = torch.from_numpy(labels[is_val]).to(device)
    print(f'[pretrain] train {len(x_train)} | val {len(x_val)} (val subjects: {n_val})')

    model = TemporalPretrainer(seq_feature_dim=sequences.shape[2]).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / max(float((y_train == 1).sum()), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(x_train))
        total_loss = 0.0
        for start in range(0, len(x_train), args.batch_size):
            batch_idx = permutation[start:start + args.batch_size]
            batch_x = x_train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)

        model.eval()
        with torch.no_grad():
            val_acc = (((torch.sigmoid(model(x_val)) >= 0.5).float()) == y_val).float().mean().item()
        print(f'ep {epoch:02d} | train_loss {total_loss / max(len(x_train), 1):.4f} | val_acc {val_acc:.4f}')

    # Save only the modules shared with TwoStreamTeacher and discard the head.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'temporal_proj': model.temporal_proj.state_dict(),
            'temporal_encoder': model.temporal_encoder.state_dict(),
        },
        out_path,
    )
    print(f'[pretrain] temporal checkpoint saved to {out_path}')


if __name__ == '__main__':
    main()
