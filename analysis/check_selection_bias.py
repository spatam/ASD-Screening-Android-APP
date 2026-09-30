"""Is selecting the "most informative" images a real gain or bias?

Selecting the K most discriminative stimuli out of 300 with 27 subjects and
then evaluating on the 28th looks correct (the ranking excludes the evaluated
subject). But the 28 leave-one-subject-out rankings differ from one another by
a single subject, so they are almost identical. In effect the selection uses
every subject. This is the classic bias of feature selection outside the
validation loop.

The test permutes the subject labels, which breaks any real link between gaze
and diagnosis, and reruns the whole procedure. Under the null hypothesis an
unbiased procedure must give AUC ~0.5. Anything above 0.5 is pure bias.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from analysis.analyze_image_budget import image_ranking_loso, load_matrix


def informative_auc(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_subj = logits.shape[0]
    scores = np.zeros(n_subj)
    for s in range(n_subj):
        order = image_ranking_loso(logits, labels, s)
        avail = set(np.flatnonzero(~np.isnan(logits[s])).tolist())
        chosen = [i for i in order if i in avail][:k]
        scores[s] = float(np.mean(logits[s, chosen]))
    return roc_auc_score(labels, scores)


def random_auc(logits: np.ndarray, labels: np.ndarray, k: int, rng) -> float:
    n_subj = logits.shape[0]
    scores = np.zeros(n_subj)
    for s in range(n_subj):
        avail = np.flatnonzero(~np.isnan(logits[s]))
        chosen = rng.choice(avail, size=min(k, len(avail)), replace=False)
        scores[s] = float(np.mean(logits[s, chosen]))
    return roc_auc_score(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--oof-csv', required=True)
    parser.add_argument('--phase', default='phase3')
    parser.add_argument('--n-perm', type=int, default=100)
    parser.add_argument('--ks', default='1,5,15,50')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    logits, labels, subj, imgs = load_matrix(Path(args.oof_csv), args.phase)
    ks = [int(x) for x in args.ks.split(',')]

    print(f'subjects={len(subj)} images={len(imgs)} permutations={args.n_perm}\n')
    print(f'{"K":>4} {"obs. AUC":>9} {"null AUC":>10} {"p":>7}   verdict')
    for k in ks:
        observed = informative_auc(logits, labels, k)
        null = np.empty(args.n_perm)
        for i in range(args.n_perm):
            null[i] = informative_auc(logits, rng.permutation(labels), k)
        p = (float((null >= observed).sum()) + 1) / (args.n_perm + 1)
        verdict = 'real signal' if p < 0.05 else 'SELECTION BIAS'
        print(f'{k:>4} {observed:>9.4f} {null.mean():>10.4f} {p:>7.3f}   {verdict}')

    print('\nFor comparison, the same check on the random strategy (no selection)')
    for k in ks:
        observed = random_auc(logits, labels, k, rng)
        null = np.empty(args.n_perm)
        for i in range(args.n_perm):
            null[i] = random_auc(logits, rng.permutation(labels), k, rng)
        p = (float((null >= observed).sum()) + 1) / (args.n_perm + 1)
        verdict = 'real signal' if p < 0.05 else 'SELECTION BIAS'
        print(f'{k:>4} {observed:>9.4f} {null.mean():>10.4f} {p:>7.3f}   {verdict}')


if __name__ == '__main__':
    main()
