from __future__ import annotations

import io
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    IMAGE_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    build_sequence,
    build_visual_tensor,
    load_stimulus,
)
from .utils import imagenet_normalize, natural_key


SUBJECTS_PER_CLASS = 14

# Variants used by the paper's ablations (Section VI.B, Table "sanity checks").
#   blend          : stimulus + real heatmap, blended with blend_alpha (main pipeline)
#   image_only     : stimulus only, no gaze information
#   random_heatmap : stimulus + surrogate heatmap sampled from the global
#                    fixation distribution, keeping the record length
INPUT_MODES = {'blend', 'image_only', 'random_heatmap'}

# Subsets of scanpath files:
#   retained  : only files with exactly 14 subjects, truncated to 14 (the paper's protocol, 2216 records)
#   all       : all files, no cap on subjects (held-out-stimulus protocol, 7296 records)
#   discarded : complement of ``retained``, for the selection-bias analysis (5080 records)
RETENTION_MODES = {'retained', 'all', 'discarded'}

# Grouping key for StratifiedGroupKFold, either the subject (original
# protocol) or the image identity (held-out-stimulus protocol).
GROUP_MODES = {'subject', 'image'}


@dataclass(frozen=True)
class HuiyuRecord:
    image_id: str
    label: int
    group: str
    slot: int
    fixations: np.ndarray
    stimulus_path: str | None


class HuiyuPerImageDataset(Dataset):
    """Huiyu/Saliency4ASD dataset, aligned with the paper.

    Each record is one (subject-slot, image) pair. To reproduce the counts in
    the paper, the ``exact_subject_count`` filter applies to each scanpath file
    of each class separately. It keeps every ASD or TD file with exactly 14
    subjects. The dataset also drops sequences with fewer than two fixations.
    """

    def __init__(
        self,
        huiyu_dir: str | Path,
        max_seq_len: int = 25,
        blend_alpha: float = 0.55,
        exact_subject_count: bool = True,
        input_mode: str = 'blend',
        retention: str | None = None,
        group_by: str = 'subject',
        max_slots: int | None = None,
        balance_per_image: bool = False,
        random_heatmap_seed: int = 42,
    ) -> None:
        self.huiyu_dir = Path(huiyu_dir).expanduser().resolve()
        if not self.huiyu_dir.exists():
            raise FileNotFoundError(
                f'Huiyu dataset not found in {self.huiyu_dir}. The paper depends on Saliency4ASD, which is missing from the current workspace.'
            )

        if input_mode not in INPUT_MODES:
            raise ValueError(f'Invalid input_mode: {input_mode!r}. Allowed: {sorted(INPUT_MODES)}')
        if group_by not in GROUP_MODES:
            raise ValueError(f'Invalid group_by: {group_by!r}. Allowed: {sorted(GROUP_MODES)}')

        # ``retention`` generalises the old boolean flag ``exact_subject_count``.
        # If the caller omits it, we derive it from that flag, so the main
        # pipeline we already reproduced behaves as before.
        if retention is None:
            retention = 'retained' if exact_subject_count else 'all'
        if retention not in RETENTION_MODES:
            raise ValueError(f'Invalid retention: {retention!r}. Allowed: {sorted(RETENTION_MODES)}')

        self.max_seq_len = max_seq_len
        self.blend_alpha = float(blend_alpha)
        self.exact_subject_count = exact_subject_count
        self.input_mode = input_mode
        self.retention = retention
        self.group_by = group_by
        # ``max_slots`` caps the number of subjects read from each file. The
        # subject_all protocol needs it. Huiyu files hold 8 to 17 subjects, and
        # without a cap the slots above 13 form tiny groups (1-18 records versus
        # ~280 in the low slots), unusable as cross-validation folds. A cap of
        # 14 gives the same 28 groups as the paper's protocol and loses only
        # 0.3% of the records.
        self.max_slots = int(max_slots) if max_slots is not None else None
        # With ``balance_per_image`` each image contributes the same number of
        # ASD and TD subjects. This cancels whatever information stimulus
        # identity still carries. If P(ASD|image)=0.5 for every image, a
        # classifier that looked only at *which* image was shown would get an
        # AUC of exactly 0.5, so any result above chance must come from gaze.
        self.balance_per_image = bool(balance_per_image)
        self.random_heatmap_seed = int(random_heatmap_seed)
        self.records: list[HuiyuRecord] = []
        self._image_index = self._index_stimuli()
        self._stimulus_cache: dict[str, np.ndarray] = {}
        self._fixation_pool: np.ndarray | None = None
        self._load_records()

    @classmethod
    def from_data_dirs(
        cls,
        data_dirs: Sequence[str | Path],
        max_seq_len: int = 25,
        blend_alpha: float = 0.55,
    ) -> 'HuiyuPerImageDataset':
        for candidate in data_dirs:
            path = Path(candidate).expanduser().resolve()
            if (path / 'TrainingData').exists() or path.name.lower().startswith('huiyu'):
                return cls(path, max_seq_len=max_seq_len, blend_alpha=blend_alpha)
        raise FileNotFoundError(
            'No Huiyu/Saliency4ASD directory found among the given paths. The public Zenodo record provides only UsageAgreement.txt, so you must add the real data by hand.'
        )

    def with_indices(self, indices: Sequence[int], augment: dict | None = None) -> 'HuiyuDatasetView':
        return HuiyuDatasetView(self, indices, augment=augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.record_to_tensors(self.records[idx], augment=None)

    def get_groups_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        groups = np.array([record.group for record in self.records])
        labels = np.array([record.label for record in self.records], dtype=np.int32)
        return groups, labels

    def record_to_tensors(self, record: HuiyuRecord, augment: dict | None = None):
        fixations = np.array(record.fixations, copy=True, dtype=np.float32)
        if augment:
            fixations = self._apply_fixation_augmentations(fixations, augment)

        seq = self._build_sequence(fixations)
        visual = self._build_visual_tensor(fixations, record)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(visual),
            torch.tensor(record.label, dtype=torch.float32),
            record.group,
        )

    def _keep_file(self, subjects: list[np.ndarray]) -> bool:
        """Decide whether a scanpath file belongs to the requested subset."""
        has_exact_14 = len(subjects) == SUBJECTS_PER_CLASS
        if self.retention == 'retained':
            return has_exact_14
        if self.retention == 'discarded':
            return not has_exact_14
        return True

    def _group_key(self, split: str, slot: int, image_id: str) -> str:
        """Key passed as ``groups`` to StratifiedGroupKFold."""
        if self.group_by == 'image':
            return f'img{image_id}'
        return f'{split}_slot{slot:02d}'

    def _load_records(self) -> None:
        retained_file_count = 0
        retained_images: set[str] = set()

        for split, label in [('ASD', 1), ('TD', 0)]:
            split_files = self._scan_class_files(split)
            for image_id in sorted(split_files, key=natural_key):
                subjects = self._parse_scanpath_file(split_files[image_id])
                if not self._keep_file(subjects):
                    continue

                retained_file_count += 1
                stimulus_path = self._find_stimulus(image_id)
                retained_images.add(image_id)

                # Truncation to 14 slots belongs to the paper's "retained"
                # protocol. The all/discarded subsets use every slot in the file
                # (8 to 17 subjects, depending on the image), unless
                # ``max_slots`` sets an explicit cap.
                if self.retention == 'retained':
                    usable = subjects[:SUBJECTS_PER_CLASS]
                elif self.max_slots is not None:
                    usable = subjects[: self.max_slots]
                else:
                    usable = subjects
                for slot, seq in enumerate(usable):
                    if len(seq) < 2:
                        continue
                    self.records.append(
                        HuiyuRecord(
                            image_id=image_id,
                            label=label,
                            group=self._group_key(split, slot, image_id),
                            slot=slot,
                            fixations=seq.astype(np.float32),
                            stimulus_path=stimulus_path,
                        )
                    )

        if self.balance_per_image:
            self._balance_records_per_image()

        if not self.records:
            raise RuntimeError(
                'No Huiyu records built. Check that the scanpath files and images of the Saliency4ASD dataset exist and are readable.'
            )

        cap = self.max_slots if self.max_slots is not None else '-'
        print(f'[Huiyu] retention={self.retention} input={self.input_mode} group_by={self.group_by} '
              f'max_slots={cap} balanced={self.balance_per_image}')
        print(f'[Huiyu] retained files: {retained_file_count}')
        print(f'[Huiyu] retained stimuli: {len(retained_images)}')
        print(f'[Huiyu] per-image records: {len(self.records)}')

    def _balance_records_per_image(self) -> None:
        """Balance the number of ASD and TD subjects for each image.

        For each image, keep ``min(n_ASD, n_TD)`` subjects per class. The choice
        favours the slots with the fewest records kept so far, so the loss
        spreads evenly and the 28 groups stay comparable. Choosing slots in
        order would empty the high slots, which are already rarer, and leave
        them as unusable CV groups.
        """
        rng = np.random.default_rng(self.random_heatmap_seed)
        by_image: dict[str, dict[int, list[int]]] = {}
        for idx, record in enumerate(self.records):
            by_image.setdefault(record.image_id, {0: [], 1: []})[record.label].append(idx)

        kept_per_group: dict[str, int] = {}
        keep: list[int] = []
        for image_id in sorted(by_image):
            per_label = by_image[image_id]
            quota = min(len(per_label[0]), len(per_label[1]))
            if quota == 0:
                continue
            for label in (0, 1):
                candidates = per_label[label]
                candidates.sort(
                    key=lambda i: (kept_per_group.get(self.records[i].group, 0), rng.random())
                )
                for idx in candidates[:quota]:
                    group = self.records[idx].group
                    kept_per_group[group] = kept_per_group.get(group, 0) + 1
                    keep.append(idx)

        keep.sort()
        self.records = [self.records[i] for i in keep]

    def _scan_class_files(self, split: str) -> dict[str, Path]:
        roots = [self.huiyu_dir / 'TrainingData' / split, self.huiyu_dir / split]
        folder = next((root for root in roots if root.exists()), None)
        if folder is None:
            raise FileNotFoundError(f'Scanpath folder {split} not found under {self.huiyu_dir}.')

        mapping: dict[str, Path] = {}
        for path in sorted(folder.glob('*.txt')):
            image_id = self._image_id_from_filename(path.name)
            mapping[image_id] = path
        return mapping

    def _index_stimuli(self) -> dict[str, Path]:
        candidates = [
            self.huiyu_dir / 'TrainingData' / 'Images',
            self.huiyu_dir / 'TrainingData' / 'Image',
            self.huiyu_dir / 'Images',
            self.huiyu_dir / 'Image',
            self.huiyu_dir / 'Stimuli',
            self.huiyu_dir / 'stimuli',
        ]

        mapping: dict[str, Path] = {}
        for folder in candidates:
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
                    continue
                stem = path.stem
                digits = ''.join(ch for ch in stem if ch.isdigit())
                keys = {stem, stem.lstrip('0') or '0'}
                if digits:
                    keys.add(digits)
                    keys.add(digits.lstrip('0') or '0')
                    keys.add(digits.zfill(4))
                for key in keys:
                    mapping.setdefault(key, path)
        return mapping

    def _find_stimulus(self, image_id: str) -> str | None:
        keys = [image_id, image_id.lstrip('0') or '0', image_id.zfill(4)]
        digits = ''.join(ch for ch in image_id if ch.isdigit())
        if digits:
            keys.extend([digits, digits.lstrip('0') or '0', digits.zfill(4)])
        for key in keys:
            if key in self._image_index:
                return str(self._image_index[key])
        return None

    def _image_id_from_filename(self, filename: str) -> str:
        stem = Path(filename).stem
        digits = ''.join(ch for ch in stem if ch.isdigit())
        return digits or stem

    def _parse_scanpath_file(self, path: Path) -> list[np.ndarray]:
        text = path.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            return []

        parsed = self._try_tabular_parse(text)
        if parsed:
            return parsed

        line_records = self._try_linewise_parse(text)
        if line_records:
            return line_records

        numbers = self._extract_numbers(text)
        record = self._numbers_to_sequence(numbers)
        return [record] if record is not None else []

    def _try_tabular_parse(self, text: str) -> list[np.ndarray]:
        try:
            dataframe = pd.read_csv(io.StringIO(text), sep=None, engine='python')
        except Exception:
            return []

        dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
        column_map = {column: column for column in dataframe.columns}
        idx_col = next((column_map[col] for col in column_map if col in {'idx', 'index'}), None)
        x_col = next((column_map[col] for col in column_map if col in {'x', 'x_pos', 'xpos'}), None)
        y_col = next((column_map[col] for col in column_map if col in {'y', 'y_pos', 'ypos'}), None)
        dur_col = next((column_map[col] for col in column_map if 'dur' in col or 'time' in col), None)
        subj_col = next((column_map[col] for col in column_map if 'subject' in col or 'participant' in col or 'viewer' in col or col == 'id'), None)

        if x_col is None or y_col is None:
            return []

        dataframe[x_col] = pd.to_numeric(dataframe[x_col], errors='coerce')
        dataframe[y_col] = pd.to_numeric(dataframe[y_col], errors='coerce')
        if idx_col is not None:
            dataframe[idx_col] = pd.to_numeric(dataframe[idx_col], errors='coerce')
        if dur_col is not None:
            dataframe[dur_col] = pd.to_numeric(dataframe[dur_col], errors='coerce').fillna(0.0)
        else:
            dataframe['__duration__'] = 1.0
            dur_col = '__duration__'
        dataframe = dataframe.dropna(subset=[x_col, y_col])
        if dataframe.empty:
            return []

        if idx_col is not None and subj_col is None:
            idx_values = dataframe[idx_col].fillna(-1).to_numpy()
            reset_points = np.flatnonzero(idx_values[1:] <= idx_values[:-1]) + 1
            boundaries = np.concatenate(([0], reset_points, [len(dataframe)]))
            out: list[np.ndarray] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                group = dataframe.iloc[start:end]
                seq = group[[x_col, y_col, dur_col]].to_numpy(dtype=np.float32)
                if len(seq):
                    out.append(seq)
            return out

        if subj_col is None:
            seq = dataframe[[x_col, y_col, dur_col]].to_numpy(dtype=np.float32)
            return [seq] if len(seq) else []

        out: list[np.ndarray] = []
        for _, group in dataframe.groupby(subj_col, sort=False):
            seq = group[[x_col, y_col, dur_col]].to_numpy(dtype=np.float32)
            if len(seq):
                out.append(seq)
        return out

    def _try_linewise_parse(self, text: str) -> list[np.ndarray]:
        sequences: list[np.ndarray] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            numbers = self._extract_numbers(line)
            record = self._numbers_to_sequence(numbers)
            if record is not None:
                sequences.append(record)
        return sequences

    def _extract_numbers(self, text: str) -> list[float]:
        return [float(token) for token in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)]

    def _numbers_to_sequence(self, numbers: list[float]) -> np.ndarray | None:
        if len(numbers) < 4:
            return None

        if len(numbers) % 4 == 0:
            array = np.asarray(numbers, dtype=np.float32).reshape(-1, 4)
            seq = array[:, 1:4]
        elif len(numbers) % 3 == 0:
            seq = np.asarray(numbers, dtype=np.float32).reshape(-1, 3)
        elif len(numbers) % 2 == 0:
            array = np.asarray(numbers, dtype=np.float32).reshape(-1, 2)
            duration = np.ones((array.shape[0], 1), dtype=np.float32)
            seq = np.concatenate([array, duration], axis=1)
        else:
            usable = len(numbers) - (len(numbers) % 3)
            if usable < 3:
                return None
            seq = np.asarray(numbers[:usable], dtype=np.float32).reshape(-1, 3)

        return seq.astype(np.float32)

    def _apply_fixation_augmentations(self, fixations: np.ndarray, augment: dict) -> np.ndarray:
        out = np.array(fixations, copy=True, dtype=np.float32)
        jitter_sigma = float(augment.get('gaze_jitter', 0.0) or 0.0)
        fix_dropout = float(augment.get('fix_dropout', 0.0) or 0.0)

        if jitter_sigma > 0.0:
            out[:, 0] += np.random.normal(0.0, jitter_sigma * SCREEN_WIDTH, size=len(out)).astype(np.float32)
            out[:, 1] += np.random.normal(0.0, jitter_sigma * SCREEN_HEIGHT, size=len(out)).astype(np.float32)
            out[:, 0] = np.clip(out[:, 0], 0.0, SCREEN_WIDTH)
            out[:, 1] = np.clip(out[:, 1], 0.0, SCREEN_HEIGHT)

        if fix_dropout > 0.0 and len(out) > 1:
            keep = np.random.rand(len(out)) >= fix_dropout
            if not np.any(keep):
                keep[np.random.randint(0, len(out))] = True
            out = out[keep]

        return out

    def _build_sequence(self, fixations: np.ndarray) -> np.ndarray:
        return build_sequence(fixations, self.max_seq_len)

    def _build_fixation_pool(self) -> np.ndarray:
        """Global pool of all fixations, used by the random-heatmap control."""
        if self._fixation_pool is None:
            self._fixation_pool = np.concatenate(
                [record.fixations for record in self.records], axis=0
            ).astype(np.float32)
        return self._fixation_pool

    def _random_fixations(self, record: HuiyuRecord, count: int) -> np.ndarray:
        """Random surrogate of ``count`` fixations drawn from the global distribution.

        It keeps the record length, so the only information it destroys is the
        gaze geometry. The seed depends on the record identity through CRC32, which
        stays the same in every process. Python's ``hash`` of a string changes with
        each interpreter start, so runs that used it could not be repeated exactly.
        """
        pool = self._build_fixation_pool()
        key = f'{record.image_id}|{record.slot}|{record.label}'.encode()
        seed = (zlib.crc32(key) ^ self.random_heatmap_seed) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        picked = rng.integers(0, len(pool), size=count)
        return pool[picked]

    def _build_visual_tensor(self, fixations: np.ndarray, record: HuiyuRecord) -> np.ndarray:
        if record.stimulus_path:
            cached = self._stimulus_cache.get(record.stimulus_path)
            if cached is None:
                cached = load_stimulus(record.stimulus_path)
                self._stimulus_cache[record.stimulus_path] = cached
            image_chw = cached
        else:
            image_chw = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # In the image-only baseline no gaze information enters the tensor.
        if self.input_mode == 'image_only':
            return imagenet_normalize(image_chw)

        if self.input_mode == 'random_heatmap':
            fixations = self._random_fixations(record, len(fixations))

        return build_visual_tensor(image_chw, fixations, self.blend_alpha)


class HuiyuDatasetView(Dataset):
    def __init__(self, base_dataset: HuiyuPerImageDataset, indices: Sequence[int], augment: dict | None = None) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        record = self.base_dataset.records[int(self.indices[idx])]
        return self.base_dataset.record_to_tensors(record, augment=self.augment)


class UnifiedGazeDataset(HuiyuPerImageDataset):
    def __init__(self, data_dirs: Sequence[str | Path] | str | Path, max_seq_len: int = 25, blend_alpha: float = 0.55):
        if isinstance(data_dirs, (str, Path)):
            chosen = Path(data_dirs)
        else:
            chosen = None
            for candidate in data_dirs:
                path = Path(candidate)
                if path.exists() and ((path / 'TrainingData').exists() or path.name.lower().startswith('huiyu')):
                    chosen = path
                    break
            if chosen is None:
                raise FileNotFoundError(
                    'UnifiedGazeDataset needs a Huiyu/Saliency4ASD directory. The Cilia and Qiao-He datasets are outside the main pipeline of the paper.'
                )
        super().__init__(chosen, max_seq_len=max_seq_len, blend_alpha=blend_alpha)
