"""Empirically test whether the "slots" of the Huiyu scanpath files match real subjects.

The raw files have no subject identifier. Only a reset of the Idx column
separates the sequences, so the slot is the ordinal position of the block in
the file. The paper's protocol assumes that slot k is the same child in every
image. Otherwise, grouping by slot would not prevent leakage between subjects.

If slots were stable subjects, some children would make systematically more or
fewer fixations than the others. The between-slot variance of per-record
statistics would then exceed the variance expected by chance. We compare the
observed between-slot variance with a null distribution built by permuting the
slot labels within each image. This destroys any subject identity and keeps
the per-image structure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from asd_gaze.dataset import HuiyuPerImageDataset


def between_slot_variance(slots: np.ndarray, values: np.ndarray) -> float:
    """Variance of the per-slot means (weighted by the number of records per slot)."""
    means = []
    weights = []
    for slot in np.unique(slots):
        mask = slots == slot
        if mask.sum() < 2:
            continue
        means.append(values[mask].mean())
        weights.append(mask.sum())
    if len(means) < 2:
        return 0.0
    means = np.asarray(means)
    weights = np.asarray(weights, dtype=float)
    grand = np.average(means, weights=weights)
    return float(np.average((means - grand) ** 2, weights=weights))


def permutation_test(
    slots: np.ndarray,
    values: np.ndarray,
    images: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    observed = between_slot_variance(slots, values)

    # For the null, permute the slots WITHIN each image. Each image keeps the
    # same set of slots and the same values, but the slot->record pairing
    # becomes random, so subject identity disappears.
    by_image: dict[str, np.ndarray] = {}
    for image in np.unique(images):
        by_image[image] = np.flatnonzero(images == image)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        permuted = slots.copy()
        for idx in by_image.values():
            permuted[idx] = rng.permutation(slots[idx])
        null[i] = between_slot_variance(permuted, values)

    p_value = float((null >= observed).sum() + 1) / (n_perm + 1)
    return observed, float(null.mean()), p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--huiyu-dir', required=True)
    parser.add_argument('--n-perm', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    for retention in ('retained', 'all'):
        dataset = HuiyuPerImageDataset(
            args.huiyu_dir,
            retention=retention,
            group_by='subject',
        )
        records = dataset.records
        print(f'\n=== retention={retention}: {len(records)} records ===')

        # Per-record statistics that do not depend on image content.
        n_fix = np.array([len(r.fixations) for r in records], dtype=float)
        mean_dur = np.array([float(r.fixations[:, 2].mean()) for r in records])
        # Spatial spread measures how much the child explores the image.
        spread = np.array([
            float(np.sqrt(r.fixations[:, 0].var() + r.fixations[:, 1].var()))
            for r in records
        ])

        slots = np.array([f'{r.group}' for r in records])
        images = np.array([r.image_id for r in records])
        # Analyze ASD and TD separately, since slots are per class.
        split = np.array([g.split('_')[0] for g in slots])

        for feature_name, values in (
            ('n_fixations', n_fix),
            ('mean_duration', mean_dur),
            ('spatial_spread', spread),
        ):
            for cls in ('ASD', 'TD'):
                mask = split == cls
                obs, null_mean, p = permutation_test(
                    slots[mask], values[mask], images[mask], args.n_perm, rng
                )
                ratio = obs / null_mean if null_mean > 0 else float('inf')
                verdict = 'SLOT INFORMATIVE' if p < 0.05 else 'slot ~ random'
                print(
                    f'  {feature_name:<15} {cls}: var_observed={obs:.4f} '
                    f'var_null={null_mean:.4f} ratio={ratio:5.2f} p={p:.4f}  -> {verdict}'
                )

        # How many slots per class, and how many records each.
        counts: dict[str, int] = defaultdict(int)
        for g in slots:
            counts[g] += 1
        print('  records per slot:')
        for g in sorted(counts):
            print(f'    {g}: {counts[g]}')


if __name__ == '__main__':
    main()
