import numpy as np
import pytest

from asd_gaze.preprocessing import (
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    build_heatmap,
    build_sequence,
    build_visual_tensor,
)


def test_sequence_scales_over_all_fixations_then_truncates():
    fixations = np.array([[x, 2 * x, 100.0] for x in range(40)], dtype=np.float32)
    seq = build_sequence(fixations)
    assert seq.shape == (MAX_SEQ_LEN, 3)
    # Scaling uses all 40 fixations, so row 24 sits at 24/39 of the range.
    assert seq[24, 0] == pytest.approx(24 / 39, abs=1e-6)
    assert seq[0, 2] == pytest.approx(np.log1p(100.0) / 10.0, abs=1e-6)


def test_sequence_pads_with_zeros_and_handles_constant_axes():
    seq = build_sequence(np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 20.0]], dtype=np.float32))
    assert np.all(seq[:2, :2] == 0.5)
    assert np.all(seq[2:] == 0.0)
    assert np.all(build_sequence(np.zeros((0, 3), dtype=np.float32)) == 0.0)


def test_heatmap_uses_screen_scale_for_stimulus_pixels():
    heatmap = build_heatmap(np.array([[1024.0, 768.0, 300.0]], dtype=np.float32))
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    # 1024 / 1280 * 223 = 178.4 and 768 / 1024 * 223 = 167.25, truncated.
    assert (x, y) == (178, 167)
    assert heatmap.max() == pytest.approx(1.0)


def test_visual_tensor_shape_and_normalisation():
    image = np.full((3, IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    visual = build_visual_tensor(image, np.zeros((0, 3), dtype=np.float32))
    assert visual.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    # Without fixations the jet map of a zero heatmap is (0, 0, 0.5).
    expected_red = (0.55 * 0.5 - 0.485) / 0.229
    assert visual[0, 0, 0] == pytest.approx(expected_red, abs=1e-5)
