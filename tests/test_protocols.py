"""Protocol invariants on a small synthetic dataset laid out like Saliency4ASD."""

import numpy as np
import pytest
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from asd_gaze.dataset import HuiyuPerImageDataset
from asd_gaze.train_common import PROTOCOL_SPECS

# subjects per (image, class): images 1-4 have 14 viewers in both classes, image 5 misses
# some ASD viewers and image 6 has 16 TD viewers, as in the real data.
LAYOUT = {1: (14, 14), 2: (14, 14), 3: (14, 14), 4: (14, 14), 5: (11, 14), 6: (14, 16)}


def write_scanpaths(path, subjects, rng):
    lines = ['Idx, x, y, duration']
    for _ in range(subjects):
        for idx in range(int(rng.integers(3, 9))):
            lines.append(f'{idx},{rng.integers(0, 1024)},{rng.integers(0, 768)},{rng.integers(8, 500)}')
    path.write_text('\n'.join(lines) + '\n')


@pytest.fixture(scope='module')
def huiyu_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp('saliency4asd')
    rng = np.random.default_rng(0)
    for folder in ('ASD', 'TD', 'Images'):
        (root / 'TrainingData' / folder).mkdir(parents=True)
    for image, (asd, td) in LAYOUT.items():
        write_scanpaths(root / 'TrainingData' / 'ASD' / f'ASD_scanpath_{image}.txt', asd, rng)
        write_scanpaths(root / 'TrainingData' / 'TD' / f'TD_scanpath_{image}.txt', td, rng)
        Image.new('RGB', (64, 48), (image * 30, 80, 120)).save(root / 'TrainingData' / 'Images' / f'{image}.png')
    return root


def build(huiyu_dir, protocol):
    spec = PROTOCOL_SPECS[protocol]
    return HuiyuPerImageDataset(
        huiyu_dir,
        retention=spec['retention'],
        group_by=spec['group_by'],
        max_slots=spec['max_slots'],
        balance_per_image=spec['balance_per_image'],
    )


def test_retained_protocol_keeps_only_exact_14_files(huiyu_dir):
    ds = build(huiyu_dir, 'subject')
    kept = {(r.image_id, r.label) for r in ds.records}
    assert ('5', 1) not in kept and ('6', 0) not in kept
    assert ('5', 0) in kept and ('6', 1) in kept
    assert all(r.slot < 14 for r in ds.records)


def test_deployment_protocol_balances_every_image(huiyu_dir):
    ds = build(huiyu_dir, 'subject_all')
    for image in LAYOUT:
        labels = [r.label for r in ds.records if r.image_id == str(image)]
        assert labels.count(0) == labels.count(1) > 0
    assert all(r.slot < 14 for r in ds.records)


@pytest.mark.parametrize('protocol', ['subject', 'subject_all', 'stimulus'])
def test_folds_never_share_a_group(huiyu_dir, protocol):
    ds = build(huiyu_dir, protocol)
    groups, labels = ds.get_groups_and_labels()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in splitter.split(np.zeros(len(ds)), labels, groups):
        assert not set(groups[train_idx]) & set(groups[val_idx])
    expected_prefix = 'img' if protocol == 'stimulus' else ('ASD_slot', 'TD_slot')
    assert all(g.startswith(expected_prefix) for g in groups)


def test_record_tensors_have_model_shapes(huiyu_dir):
    seq, visual, label, group = build(huiyu_dir, 'subject')[0]
    assert tuple(seq.shape) == (25, 3)
    assert tuple(visual.shape) == (3, 224, 224)
    assert float(label) in (0.0, 1.0)
