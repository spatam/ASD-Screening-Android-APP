import numpy as np
import pytest

from asd_gaze.utils import aggregate_subject_probs


def test_logit_space_is_the_default():
    labels, scores = aggregate_subject_probs(['a', 'a'], [1, 1], [0.9, 0.6])
    assert labels.tolist() == [1]
    assert scores[0] == pytest.approx(0.7861, abs=1e-4)


def test_probability_space_on_request():
    _, scores = aggregate_subject_probs(['a', 'a'], [1, 1], [0.9, 0.6], space='prob')
    assert scores[0] == pytest.approx(0.75)


def test_groups_are_sorted_and_invalid_space_fails():
    labels, scores = aggregate_subject_probs(['b', 'a'], [0, 1], [0.2, 0.8])
    assert labels.tolist() == [1, 0]
    assert np.all((scores > 0) & (scores < 1))
    with pytest.raises(ValueError):
        aggregate_subject_probs(['a'], [1], [0.5], space='median')
