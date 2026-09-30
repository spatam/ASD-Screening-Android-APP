"""Regenerate the paper's summary figures from the exact, audited table values.

Every number here is copied from the verified tables of the paper
(tab:results_main, tab:folds, tab:clf, tab:subject_all). No run directory reproduces
the published numbers bit for bit, so the script does not read any.
Consolidates what used to be four figures (fold_auc, classification, summary,
distill_curve) into three, each covering both the retained-subject and the
deployment protocol side by side.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PHASES = ['Teacher\n(Ph.1)', 'Teacher+aug\n(Ph.2)', 'Student\n(Ph.3)']
COLORS = {'retained': '#1f77b4', 'deployment': '#d62728'}

RETAINED = dict(
    oof_auc=[0.9867, 0.9874, 0.9894],
    pr_auc=[0.9801, 0.9810, 0.9841],
    acc=[0.9310, 0.9404, 0.9346],
    prec=[0.9003, 0.9196, 0.8830],
    rec=[0.9225, 0.9261, 0.9566],
    spec=[0.9358, 0.9489, 0.9192],
    f1=[0.9113, 0.9228, 0.9183],
    folds=[
        [0.9914, 0.9859, 0.9840, 0.9871, 0.9847],
        [0.9921, 0.9876, 0.9851, 0.9900, 0.9834],
        [0.9892, 0.9895, 0.9881, 0.9915, 0.9909],
    ],
    params_m=[85.95, 85.95, 1.6],
)

DEPLOYMENT = dict(
    oof_auc=[0.6600, 0.6212, 0.6859],
    pr_auc=[0.6470, 0.6032, 0.6726],
    acc=[0.6192, 0.5968, 0.6446],
    prec=[0.6157, 0.5925, 0.6616],
    rec=[0.6343, 0.6195, 0.5920],
    spec=[0.6034, 0.5730, 0.6912],
    f1=[0.6249, 0.6057, 0.6249],
    folds=[
        [0.649, 0.688, 0.663, 0.651, 0.663],
        [0.607, 0.629, 0.587, 0.625, 0.626],
        [0.673, 0.707, 0.693, 0.696, 0.670],
    ],
    params_m=[85.95, 85.95, 1.6],
)


def fig_summary(out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    x = np.arange(3)
    w = 0.35
    ax1.bar(x - w / 2, RETAINED['oof_auc'], w, label='Retained-subject', color=COLORS['retained'])
    ax1.bar(x + w / 2, DEPLOYMENT['oof_auc'], w, label='Deployment', color=COLORS['deployment'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Ph.1', 'Ph.2', 'Ph.3'])
    ax1.set_ylabel('OOF per-image AUC')
    ax1.set_ylim(0.5, 1.02)
    ax1.axhline(0.5, ls=':', c='gray', lw=1)
    ax1.set_title('AUC by phase and protocol')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(alpha=0.3, ls='--', axis='y')

    params = [RETAINED['params_m'][0], RETAINED['params_m'][2]]
    aucs = [RETAINED['oof_auc'][0], RETAINED['oof_auc'][2]]
    labels = ['Teacher', 'Student']
    ax2.scatter(params, aucs, s=90, c=['#7f7f7f', COLORS['retained']], zorder=5)
    for p, a, lbl in zip(params, aucs, labels):
        ax2.annotate(f'{lbl}\n{a:.3f} AUC, {p:g}M', (p, a), textcoords='offset points',
                     xytext=(8, -4), fontsize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Total parameters (millions, log scale)')
    ax2.set_ylabel('OOF per-image AUC (retained-subject)')
    ax2.set_ylim(0.97, 1.0)
    ax2.set_title('Accuracy vs. model footprint')
    ax2.grid(alpha=0.3, ls='--')

    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def fig_fold_auc(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
    fold_x = np.arange(1, 6)
    phase_labels = ['Ph.1 Teacher', 'Ph.2 Teacher+aug', 'Ph.3 Student']
    markers = ['o', 's', '^']

    for i, lbl in enumerate(phase_labels):
        axes[0].plot(fold_x, RETAINED['folds'][i], marker=markers[i], label=lbl)
    axes[0].set_title('Retained-subject protocol')
    axes[0].set_ylim(0.975, 1.0)

    for i, lbl in enumerate(phase_labels):
        axes[1].plot(fold_x, DEPLOYMENT['folds'][i], marker=markers[i], label=lbl)
    axes[1].set_title('Deployment protocol')
    axes[1].set_ylim(0.55, 0.75)

    for ax in axes:
        ax.set_xlabel('Fold')
        ax.set_ylabel('Per-fold AUC')
        ax.set_xticks(fold_x)
        ax.grid(alpha=0.3, ls='--')
        ax.legend(fontsize=7.5, loc='lower right')

    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def fig_classification(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    metrics = ['acc', 'prec', 'rec', 'spec', 'f1']
    metric_labels = ['Acc', 'Prec', 'Rec/Sens', 'Spec', 'F1']
    x = np.arange(len(metrics))
    w = 0.25
    phase_colors = ['#4c72b0', '#dd8452', '#55a868']

    for i, phase_lbl in enumerate(['Ph.1', 'Ph.2', 'Ph.3']):
        vals = [RETAINED[m][i] for m in metrics]
        axes[0].bar(x + (i - 1) * w, vals, w, label=phase_lbl, color=phase_colors[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_title('Retained-subject (Youden threshold)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, ls='--', axis='y')

    for i, phase_lbl in enumerate(['Ph.1', 'Ph.2', 'Ph.3']):
        vals = [DEPLOYMENT[m][i] for m in metrics]
        axes[1].bar(x + (i - 1) * w, vals, w, label=phase_lbl, color=phase_colors[i])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_labels)
    axes[1].set_ylim(0.5, 0.75)
    axes[1].set_title('Deployment (Youden threshold)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, ls='--', axis='y')

    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    fig_summary(out_dir / 'fig_summary_modified.pdf')
    fig_fold_auc(out_dir / 'fig_fold_auc.pdf')
    fig_classification(out_dir / 'fig_classification.pdf')


if __name__ == '__main__':
    main()
