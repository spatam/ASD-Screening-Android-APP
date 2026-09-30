import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from asd_gaze.export_onnx import export_single
from asd_gaze.student_model import StudentASD

REPO = Path(__file__).resolve().parents[1]
CHECKSUMS = [line.split() for line in (REPO / 'models' / 'SHA256SUMS').read_text().splitlines() if line.strip()]


@pytest.mark.parametrize('digest,relpath', CHECKSUMS, ids=[c[1] for c in CHECKSUMS])
def test_released_model_checksum_and_contract(digest, relpath):
    path = REPO / relpath
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    assert [i.name for i in session.get_inputs()] == ['gaze_seq', 'stimulus_image']
    rng = np.random.default_rng(0)
    out = session.run(['logit'], {
        'gaze_seq': rng.random((1, 25, 3), dtype=np.float32),
        'stimulus_image': rng.standard_normal((1, 3, 224, 224), dtype=np.float32),
    })[0]
    assert out.size == 1 and np.isfinite(out).all()


def test_student_size_matches_the_paper():
    model = StudentASD(pretrained=False)
    assert sum(p.numel() for p in model.parameters()) == 1_602_625


def test_export_uses_the_app_input_names(tmp_path):
    torch.manual_seed(0)
    ckpt = tmp_path / 'student.pth'
    torch.save(StudentASD(pretrained=False).eval().state_dict(), ckpt)
    out = tmp_path / 'student.onnx'
    export_single(ckpt, out, opset=18, max_seq_len=25, hidden=64, verify=True)
    session = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    assert [i.name for i in session.get_inputs()] == ['gaze_seq', 'stimulus_image']
    assert [o.name for o in session.get_outputs()] == ['logit']
