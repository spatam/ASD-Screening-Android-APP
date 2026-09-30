"""Stimulus-budget curve figure for the deployment protocol.

The left panel shows subject-level AUC against the number of images shown,
with the 5--95 percentile band over random draws and the three operating modes
highlighted. The right panel shows accuracy, sensitivity and specificity on the
same scale. The top axis gives session duration, the quantity that matters
when the subject is a child.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SECONDS_PER_IMAGE = 4.0
MODES = {20: 'screening (1.3 min)', 40: 'standard (2.7 min)', 100: 'extended (6.7 min)'}


def load(curve_csv: Path, strategy: str):
    rows = [r for r in csv.DictReader(curve_csv.open()) if r['strategy'] == strategy]
    rows.sort(key=lambda r: int(r['k']))
    return {
        'k': [int(r['k']) for r in rows],
        'auc': [float(r['auc_mean']) for r in rows],
        'lo': [float(r['auc_p05']) for r in rows],
        'hi': [float(r['auc_p95']) for r in rows],
        'acc': [float(r['accuracy_mean']) for r in rows],
        'sens': [float(r['sensitivity_mean']) for r in rows],
        'spec': [float(r['specificity_mean']) for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--curve-csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    d = load(Path(args.curve_csv), 'random')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- left panel (subject-level AUC) ---
    fill = ax1.fill_between(d['k'], d['lo'], d['hi'], alpha=0.18, color='#1f77b4',
                            label='5--95 percentile over draws')
    line_auc, = ax1.plot(d['k'], d['auc'], 'o-', color='#1f77b4', lw=2, ms=4.5,
                         label='Subject-level AUC')
    ax1.axhline(0.5, ls=':', c='gray', lw=1.2)
    ax1.text(1.15, 0.512, 'stimulus identity floor (0.500)', fontsize=8, color='gray')

    # Stagger the labels, since annotations at nearby K would overlap.
    offsets = {20: (-72, -26), 40: (-30, -40), 100: (-18, -54)}
    for k, lbl in MODES.items():
        auc = d['auc'][d['k'].index(k)]
        ax1.plot([k], [auc], marker='*', ms=15, color='#d62728', zorder=5)
        ax1.annotate(lbl, (k, auc), textcoords='offset points', xytext=offsets[k],
                     fontsize=7.5, color='#d62728',
                     arrowprops=dict(arrowstyle='-', color='#d62728', lw=0.6, alpha=0.6))

    ax1.set_xscale('log')
    ax1.set_xlabel('Number of stimuli shown ($K$)')
    ax1.set_ylabel('Subject-level AUC')
    ax1.set_ylim(0.45, 1.02)
    ax1.grid(alpha=0.3, ls='--')

    top = ax1.secondary_xaxis('top', functions=(lambda x: x * SECONDS_PER_IMAGE / 60.0,
                                                lambda x: x * 60.0 / SECONDS_PER_IMAGE))
    top.set_xlabel('Session duration (minutes)', fontsize=9)

    # --- right panel (metrics at the leave-one-subject-out threshold) ---
    line_acc, = ax2.plot(d['k'], d['acc'], 'o-', color='#1f77b4', lw=2, ms=4, label='Accuracy')
    line_sens, = ax2.plot(d['k'], d['sens'], 's-', color='#2ca02c', lw=2, ms=4, label='Sensitivity')
    line_spec, = ax2.plot(d['k'], d['spec'], '^-', color='#ff7f0e', lw=2, ms=4, label='Specificity')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of stimuli shown ($K$)')
    ax2.set_ylabel('Score at leave-one-subject-out threshold')
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(alpha=0.3, ls='--')
    handles = [line_auc, fill, line_acc, line_sens, line_spec]
    labels = ['Subject-level AUC', '5--95 percentile over draws',
              'Accuracy', 'Sensitivity', 'Specificity']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False, fontsize=8, columnspacing=1.2,
               handlelength=1.5)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.94,
                        wspace=0.28)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
