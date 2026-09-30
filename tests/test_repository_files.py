import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_golden_file_is_up_to_date():
    golden = load(REPO / 'scripts' / 'make_parity_golden.py')
    problems = golden.differences(golden.GOLDEN.read_text(), golden.render())
    assert not problems, f'{problems}: run python scripts/make_parity_golden.py'


def test_stimulus_manifest_lists_300_images():
    fetch = load(REPO / 'scripts' / 'fetch_saliency4asd.py')
    hashes = fetch.expected_hashes()
    assert sorted(hashes) == list(range(1, 301))
    assert all(len(h) == 64 for h in hashes.values())
