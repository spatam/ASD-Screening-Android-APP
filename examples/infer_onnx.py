"""Run the released ONNX students on gaze data, the way the Android app does.

Single picture, fixations from a CSV file (x, y, duration_ms in stimulus pixels, one per row):

    python examples/infer_onnx.py image --stimulus 12.png --fixations fixations.csv

Simulated screening session of one Saliency4ASD child (K random pictures, session score as the
sigmoid of the mean per-image logit, compared with the app's study threshold):

    python examples/infer_onnx.py session --dataset data/saliency4asd --subject ASD_slot03 --images 40

The deployment models in the app were trained on all 28 Saliency4ASD children, so a session built
from those children only checks that the pipeline runs. It says nothing about accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from asd_gaze.preprocessing import build_sequence, build_visual_tensor, load_stimulus  # noqa: E402

DEFAULT_MODELS = REPO / 'app' / 'src' / 'main' / 'assets' / 'models'
STUDY_THRESHOLD = 0.4922  # same as ScoreAggregator.STUDY_THRESHOLD in the app


class Ensemble:
    """The five fold students; the prediction is the mean of their logits."""

    def __init__(self, models_dir: Path) -> None:
        paths = sorted(models_dir.glob('student_fold*.onnx'))
        if not paths:
            sys.exit(f'No student_fold*.onnx files in {models_dir}')
        self.sessions = [ort.InferenceSession(str(p), providers=['CPUExecutionProvider']) for p in paths]

    def logit(self, fixations: np.ndarray, stimulus_chw: np.ndarray) -> float:
        feeds = {
            'gaze_seq': build_sequence(fixations)[None],
            'stimulus_image': build_visual_tensor(stimulus_chw, fixations)[None],
        }
        return float(np.mean([s.run(['logit'], feeds)[0].reshape(-1)[0] for s in self.sessions]))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def read_fixations(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = [p for p in line.replace(',', ' ').split() if p]
        try:
            rows.append([float(v) for v in parts[:3]])
        except ValueError:
            continue  # header line
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def run_image(args: argparse.Namespace) -> None:
    ensemble = Ensemble(args.models)
    fixations = read_fixations(args.fixations)
    logit = ensemble.logit(fixations, load_stimulus(args.stimulus))
    print(f'{len(fixations)} fixations | logit {logit:+.3f} | p(ASD) {sigmoid(logit):.3f}')


def run_session(args: argparse.Namespace) -> None:
    from asd_gaze.dataset import HuiyuPerImageDataset

    dataset = HuiyuPerImageDataset(args.dataset, retention='all', group_by='subject', max_slots=14)
    records = [r for r in dataset.records if r.group == args.subject]
    if not records:
        groups = sorted({r.group for r in dataset.records})
        sys.exit(f'Unknown subject {args.subject!r}. Choose one of: {", ".join(groups)}')
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(records), size=min(args.images, len(records)), replace=False)

    ensemble = Ensemble(args.models)
    logits = []
    for i in chosen:
        record = records[i]
        logits.append(ensemble.logit(record.fixations, load_stimulus(record.stimulus_path)))
        print(f'image {record.image_id:>3} | p(ASD) {sigmoid(logits[-1]):.3f}')
    score = sigmoid(float(np.mean(logits)))
    side = 'above' if score >= STUDY_THRESHOLD else 'below'
    label = 'ASD' if records[0].label == 1 else 'TD'
    print(f'\n{args.subject} ({label}): {len(logits)} images, session score {score:.3f}, '
          f'{side} the study threshold {STUDY_THRESHOLD}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS,
                        help='folder with student_fold1..5.onnx (default: the app models)')
    sub = parser.add_subparsers(dest='mode', required=True)

    image = sub.add_parser('image', help='score one picture')
    image.add_argument('--stimulus', type=Path, required=True)
    image.add_argument('--fixations', type=Path, required=True, help='CSV: x, y, duration_ms')
    image.set_defaults(func=run_image)

    session = sub.add_parser('session', help='simulate a screening session of a dataset child')
    session.add_argument('--dataset', type=Path, required=True, help='Saliency4ASD folder')
    session.add_argument('--subject', default='ASD_slot00', help='for example ASD_slot03 or TD_slot10')
    session.add_argument('--images', type=int, default=40)
    session.add_argument('--seed', type=int, default=0)
    session.set_defaults(func=run_session)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
