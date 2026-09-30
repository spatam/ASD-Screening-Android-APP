"""How many images must the app show to decide ASD/TD for a new child?

The app shows a fixed set of stimuli to an unseen child, collects one
probability per image and combines them into a single subject-level decision.
This script simulates that exact process on the out-of-fold predictions of the
subject_all protocol. It measures how decision quality grows with the number K
of images shown, to set the paper's three usage modes: ``screening``,
``standard`` and ``extended``.

Method choices
--------------
* Aggregation in logit space (mean of the log-odds). This is the natural way
  to combine independent evidence, and it does not saturate. A mean of
  probabilities does.
* Leave-one-subject-out threshold and calibration. Classifying subject j uses
  only the other 27 subjects, so the held-out subject does not contaminate
  accuracy, sensitivity, specificity or the calibrated probabilities.
* Two strategies for choosing stimuli.
    - ``random`` repeats a random draw of K images multiple times. It gives an
      unbiased estimate of what happens with an arbitrary choice of stimuli.
    - ``informative`` takes the K most discriminative images, also ranked
      leave-one-subject-out (the ranking excludes the evaluated subject). It
      shows whether selecting stimuli works better than picking them at
      random.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Each stimulus takes 3 s on screen plus a 1 s gray interval, as in the
# original acquisition protocol that the app reproduces.
SECONDS_PER_IMAGE = 4.0
EPS = 1e-6


def load_matrix(csv_path: Path, phase: str):
    """[subjects x images] matrix of logits, with NaN where a record is missing."""
    subjects: dict[str, int] = {}
    images: dict[str, int] = {}
    rows = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            subjects.setdefault(row['subject'], len(subjects))
            images.setdefault(row['image_id'], len(images))

    subj_names = sorted(subjects, key=lambda s: subjects[s])
    img_names = sorted(images, key=lambda s: images[s])
    logits = np.full((len(subj_names), len(img_names)), np.nan)
    labels = np.full(len(subj_names), -1, dtype=int)

    col = f'prob_{phase}'
    for row in rows:
        si = subjects[row['subject']]
        ii = images[row['image_id']]
        p = float(row[col])
        p = min(max(p, EPS), 1.0 - EPS)
        logits[si, ii] = np.log(p / (1.0 - p))
        labels[si] = int(row['label'])
    return logits, labels, subj_names, img_names


def loso_threshold_metrics(scores: np.ndarray, labels: np.ndarray):
    """Accuracy/sensitivity/specificity with a leave-one-subject-out threshold."""
    preds = np.zeros(len(scores), dtype=int)
    for j in range(len(scores)):
        mask = np.ones(len(scores), dtype=bool)
        mask[j] = False
        s_tr, y_tr = scores[mask], labels[mask]
        # Youden threshold on the 27 training subjects.
        best_thr, best_score = 0.0, -np.inf
        for thr in np.unique(s_tr):
            pred = (s_tr >= thr).astype(int)
            tp = np.sum((pred == 1) & (y_tr == 1))
            tn = np.sum((pred == 0) & (y_tr == 0))
            fp = np.sum((pred == 1) & (y_tr == 0))
            fn = np.sum((pred == 0) & (y_tr == 1))
            j_stat = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1.0
            if j_stat > best_score:
                best_score, best_thr = j_stat, thr
        preds[j] = int(scores[j] >= best_thr)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    return {
        'accuracy': (tp + tn) / len(labels),
        'sensitivity': tp / max(tp + fn, 1),
        'specificity': tn / max(tn + fp, 1),
    }


def loso_calibration(scores: np.ndarray, labels: np.ndarray):
    """Brier score of probabilities calibrated by leave-one-subject-out Platt scaling."""
    probs = np.zeros(len(scores))
    for j in range(len(scores)):
        mask = np.ones(len(scores), dtype=bool)
        mask[j] = False
        y_tr = labels[mask]
        if len(np.unique(y_tr)) < 2:
            probs[j] = 0.5
            continue
        model = LogisticRegression(C=1.0, solver='lbfgs')
        model.fit(scores[mask].reshape(-1, 1), y_tr)
        probs[j] = model.predict_proba(scores[j].reshape(1, -1))[0, 1]
    return float(np.mean((probs - labels) ** 2)), probs


def standardize_loso(logits: np.ndarray) -> np.ndarray:
    """Z-score each image using the other subjects (leave-one-subject-out).

    Stimuli differ in difficulty. Some images give high logits for everyone,
    others low logits for everyone. Without normalization a subject's score
    depends on *which* images the subject happened to get, and this hurts
    short sessions most. Standardizing each image on the distribution of the
    other subjects removes that bias without using labels, so it adds no
    leakage.
    """
    out = np.full_like(logits, np.nan)
    n_subj = logits.shape[0]
    for s in range(n_subj):
        mask = np.ones(n_subj, dtype=bool)
        mask[s] = False
        others = logits[mask]
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(others, axis=0)
            std = np.nanstd(others, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        out[s] = (logits[s] - mean) / std
    return out


def image_ranking_loso(logits: np.ndarray, labels: np.ndarray, held_out: int) -> np.ndarray:
    """Rank images by discriminative power, leaving one subject out.

    The criterion is the difference in mean logit between ASD and TD subjects
    on the same image, divided by the standard deviation. This effect size
    (Cohen's d) is robust to the number of subjects available.
    """
    mask = np.ones(logits.shape[0], dtype=bool)
    mask[held_out] = False
    sub = logits[mask]
    y = labels[mask]

    scores = np.full(logits.shape[1], -np.inf)
    for i in range(logits.shape[1]):
        col = sub[:, i]
        valid = ~np.isnan(col)
        a = col[valid & (y == 1)]
        b = col[valid & (y == 0)]
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        if pooled < 1e-9:
            continue
        scores[i] = (a.mean() - b.mean()) / pooled
    return np.argsort(-scores)


def image_weights_loso(logits: np.ndarray, labels: np.ndarray, held_out: int) -> np.ndarray:
    """Weight of each image = its effect size, estimated without the held-out subject.

    The unweighted mean treats a highly discriminative stimulus and a useless
    one as equal. Adding weak stimuli brings noise without signal and can
    worsen the decision. With effect-size weights (Cohen's d), the optimal
    weights of a linear combination of independent evidence, weakly
    informative stimuli contribute little and a larger K can no longer harm
    the aggregate.
    """
    mask = np.ones(logits.shape[0], dtype=bool)
    mask[held_out] = False
    sub, y = logits[mask], labels[mask]

    weights = np.zeros(logits.shape[1])
    for i in range(logits.shape[1]):
        col = sub[:, i]
        valid = ~np.isnan(col)
        a, b = col[valid & (y == 1)], col[valid & (y == 0)]
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        if pooled >= 1e-9:
            weights[i] = (a.mean() - b.mean()) / pooled
    return weights


def evaluate_k(logits: np.ndarray, labels: np.ndarray, k: int, strategy: str,
               n_boot: int, rng: np.random.Generator, rankings: list[np.ndarray] | None,
               weights: list[np.ndarray] | None = None):
    """Subject-level metrics when showing K images, averaged over repetitions."""
    n_subj, n_img = logits.shape
    available = [np.flatnonzero(~np.isnan(logits[s])) for s in range(n_subj)]

    reps = n_boot if strategy == 'random' else 1
    aucs, accs, sens, specs, briers, effective = [], [], [], [], [], []

    for _ in range(reps):
        scores = np.zeros(n_subj)
        for s in range(n_subj):
            avail = available[s]
            if strategy == 'random':
                take = min(k, len(avail))
                chosen = rng.choice(avail, size=take, replace=False)
            else:
                # Ranking computed without subject s. Walk the global order
                # and keep the first K images that s actually has.
                order = rankings[s]
                avail_set = set(avail.tolist())
                chosen = [i for i in order if i in avail_set][:k]
                chosen = np.asarray(chosen, dtype=int)
            effective.append(len(chosen))
            if weights is None:
                scores[s] = float(np.mean(logits[s, chosen]))
            else:
                w = weights[s][chosen]
                denom = float(np.abs(w).sum())
                scores[s] = float(np.dot(w, logits[s, chosen]) / denom) if denom > 1e-9 else 0.0

        aucs.append(roc_auc_score(labels, scores))
        m = loso_threshold_metrics(scores, labels)
        accs.append(m['accuracy'])
        sens.append(m['sensitivity'])
        specs.append(m['specificity'])
        briers.append(loso_calibration(scores, labels)[0])

    return {
        'k': k,
        'strategy': strategy,
        'session_seconds': k * SECONDS_PER_IMAGE,
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'auc_p05': float(np.percentile(aucs, 5)) if len(aucs) > 1 else float(aucs[0]),
        'auc_p95': float(np.percentile(aucs, 95)) if len(aucs) > 1 else float(aucs[0]),
        'accuracy_mean': float(np.mean(accs)),
        'sensitivity_mean': float(np.mean(sens)),
        'specificity_mean': float(np.mean(specs)),
        'brier_mean': float(np.mean(briers)),
        'effective_k_mean': float(np.mean(effective)),
    }


def pick_modes(rows: list[dict]) -> dict:
    """Pick K for the three modes from the fraction of the plateau reached."""
    plateau = max(r['auc_mean'] for r in rows)
    targets = {'screening': 0.95, 'standard': 0.98, 'extended': 0.995}
    out = {}
    for name, frac in targets.items():
        hit = next((r for r in rows if r['auc_mean'] >= frac * plateau), rows[-1])
        out[name] = {
            'k': hit['k'],
            'session_seconds': hit['session_seconds'],
            'session_minutes': round(hit['session_seconds'] / 60.0, 1),
            'auc': round(hit['auc_mean'], 4),
            'accuracy': round(hit['accuracy_mean'], 4),
            'sensitivity': round(hit['sensitivity_mean'], 4),
            'specificity': round(hit['specificity_mean'], 4),
            'brier': round(hit['brier_mean'], 4),
            'target_fraction_of_plateau': frac,
        }
    out['plateau_auc'] = round(plateau, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof-csv', required=True)
    parser.add_argument('--phase', default='phase3')
    parser.add_argument('--n-boot', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--standardize', action='store_true',
                        help='Per-image z-score (leave-one-subject-out) before aggregating.')
    parser.add_argument('--out-prefix', required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    logits, labels, subj_names, img_names = load_matrix(Path(args.oof_csv), args.phase)
    if args.standardize:
        print('per-image standardization (LOSO) enabled')
        logits = standardize_loso(logits)
    avail = (~np.isnan(logits)).sum(axis=1)
    print(f'subjects={len(subj_names)} (ASD={int((labels==1).sum())}, TD={int((labels==0).sum())}) '
          f'| images={len(img_names)}')
    print(f'images per subject: min={avail.min()} median={int(np.median(avail))} max={avail.max()}')

    grid = [k for k in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100,
                        125, 150, 200, 250, 300] if k <= int(avail.max())]

    rankings = [image_ranking_loso(logits, labels, s) for s in range(len(subj_names))]
    weights = [image_weights_loso(logits, labels, s) for s in range(len(subj_names))]

    all_rows: list[dict] = []
    strategies = ('random', 'informative', 'random_weighted', 'informative_weighted')
    for strategy in strategies:
        base = strategy.replace('_weighted', '')
        weighted = strategy.endswith('_weighted')
        print(f'\n--- strategy: {strategy} ---')
        print(f'{"K":>4} {"duration":>8} {"AUC":>7} {"Acc":>7} {"Sens":>7} {"Spec":>7} {"Brier":>7}')
        for k in grid:
            row = evaluate_k(logits, labels, k, base, args.n_boot, rng,
                             rankings if base == 'informative' else None,
                             weights=weights if weighted else None)
            row['strategy'] = strategy
            all_rows.append(row)
            print(f'{k:>4} {row["session_seconds"]/60:>6.1f}m {row["auc_mean"]:>7.4f} '
                  f'{row["accuracy_mean"]:>7.4f} {row["sensitivity_mean"]:>7.4f} '
                  f'{row["specificity_mean"]:>7.4f} {row["brier_mean"]:>7.4f}')

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_name(prefix.name + '_curve.csv')
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    modes = {}
    for strategy in strategies:
        rows = [r for r in all_rows if r['strategy'] == strategy]
        modes[strategy] = pick_modes(rows)
    json_path = prefix.with_name(prefix.name + '_modes.json')
    json_path.write_text(json.dumps(modes, indent=2))

    print('\n=== Operating modes ===')
    for strategy, m in modes.items():
        print(f'\n[{strategy}] plateau AUC={m["plateau_auc"]}')
        for name in ('screening', 'standard', 'extended'):
            d = m[name]
            print(f'  {name:<13} K={d["k"]:>3} ({d["session_minutes"]:>4.1f} min) '
                  f'AUC={d["auc"]:.4f} Acc={d["accuracy"]:.4f} '
                  f'Sens={d["sensitivity"]:.4f} Spec={d["specificity"]:.4f}')
    print(f'\nWrote {csv_path} and {json_path}')


if __name__ == '__main__':
    main()
