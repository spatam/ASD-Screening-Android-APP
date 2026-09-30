"""Download the Saliency4ASD training set from Zenodo and prepare it for this repository.

    python scripts/fetch_saliency4asd.py                       # data/saliency4asd
    python scripts/fetch_saliency4asd.py --install-app-stimuli # also fill the app assets

The archive (TrainingDataset.rar, 835 MB, record 10.5281/zenodo.2647418) is checked against the
MD5 that Zenodo publishes and extracted with bsdtar, unar, 7z or unrar, whichever is installed.
The 300 stimuli are then checked against app/src/main/assets/stimuli_sha256.txt.

The dataset is by Duan et al. (2019). Its pictures come from MIT1003 and remain the property of
their authors: use them for research and do not redistribute them, inside an APK or otherwise.
Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD_API = 'https://zenodo.org/api/records/2647418'
ARCHIVE = 'TrainingDataset.rar'
# Used when the Zenodo API cannot be reached.
FALLBACK_URL = 'https://zenodo.org/records/2647418/files/TrainingDataset.rar?download=1'
FALLBACK_MD5 = '5ac8ddf53f858afea794bda97efb7f67'
MANIFEST = REPO / 'app' / 'src' / 'main' / 'assets' / 'stimuli_sha256.txt'
APP_STIMULI = REPO / 'app' / 'src' / 'main' / 'assets' / 'stimuli'
CHUNK = 1 << 20


def archive_source() -> tuple[str, str]:
    """Download URL and MD5 of the archive, read from the Zenodo API."""
    try:
        with urllib.request.urlopen(RECORD_API, timeout=30) as response:
            record = json.load(response)
        entry = next(f for f in record['files'] if f['key'] == ARCHIVE)
        return entry['links']['self'], entry['checksum'].split(':', 1)[1]
    except Exception as error:  # noqa: BLE001 - any failure falls back to the known values
        print(f'Zenodo API unavailable ({error}); using the known URL and checksum.')
        return FALLBACK_URL, FALLBACK_MD5


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_md5: str) -> None:
    if target.exists() and md5sum(target) == expected_md5:
        print(f'{target} already downloaded and verified.')
        return
    partial = target.with_suffix(target.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            offset = 0  # the server ignored the range: start again
        total = offset + int(response.headers.get('Content-Length', 0))
        with partial.open('ab' if offset else 'wb') as handle:
            done = offset
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB', end='', flush=True)
    print()
    if md5sum(partial) != expected_md5:
        partial.unlink()
        sys.exit('MD5 mismatch: the download is corrupt. Run the script again.')
    partial.rename(target)


def extract(archive: Path, dest: Path) -> None:
    commands = [
        ['bsdtar', '-xf', str(archive), '-C', str(dest)],
        ['unar', '-quiet', '-force-overwrite', '-output-directory', str(dest), str(archive)],
        ['7zz', 'x', '-y', f'-o{dest}', str(archive)],
        ['7z', 'x', '-y', f'-o{dest}', str(archive)],
        ['unrar', 'x', '-o+', str(archive), str(dest) + '/'],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        print(f'Extracting with {command[0]}...')
        if subprocess.run(command).returncode == 0:
            return
    sys.exit('Cannot extract the RAR archive. Install libarchive (bsdtar), unar, 7-Zip or unrar.')


def find_training_dir(dest: Path) -> Path:
    for candidate in sorted(dest.rglob('TrainingData')):
        if (candidate / 'Images').is_dir() and (candidate / 'ASD').is_dir():
            return candidate
    sys.exit(f'No TrainingData folder with Images/ and ASD/ found under {dest}.')


def expected_hashes() -> dict[int, str]:
    hashes = {}
    for line in MANIFEST.read_text().splitlines():
        digest, name = line.split()
        hashes[int(Path(name).stem)] = digest
    return hashes


def verify_stimuli(images: Path) -> None:
    bad = [
        index for index, digest in expected_hashes().items()
        if not (images / f'{index}.png').is_file()
        or hashlib.sha256((images / f'{index}.png').read_bytes()).hexdigest() != digest
    ]
    if bad:
        sys.exit(f'{len(bad)} stimuli are missing or differ from the expected checksums (first: {bad[:5]}).')
    print('All 300 stimuli match their SHA-256 checksums.')


def install_app_stimuli(images: Path) -> None:
    APP_STIMULI.mkdir(parents=True, exist_ok=True)
    for index in range(1, 301):
        shutil.copyfile(images / f'{index}.png', APP_STIMULI / f'{index:04d}.png')
    print(f'Copied the stimuli to {APP_STIMULI.relative_to(REPO)} (git ignores this folder).')


def main() -> None:
    parser = argparse.ArgumentParser(description='Download and prepare the Saliency4ASD training set.')
    parser.add_argument('--dest', type=Path, default=REPO / 'data' / 'saliency4asd',
                        help='where to store the archive and the extracted files')
    parser.add_argument('--archive', type=Path, help='use an archive you already downloaded')
    parser.add_argument('--install-app-stimuli', action='store_true',
                        help='copy the 300 stimuli into the app assets for a local build')
    parser.add_argument('--keep-archive', action='store_true', help='keep the .rar after extraction')
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    if archive is None:
        url, md5 = archive_source()
        archive = dest / ARCHIVE
        print(f'Downloading {ARCHIVE} (835 MB) from Zenodo...')
        download(url, archive, md5)

    extract(archive, dest)
    training = find_training_dir(dest)
    verify_stimuli(training / 'Images')
    if args.install_app_stimuli:
        install_app_stimuli(training / 'Images')
    if args.archive is None and not args.keep_archive:
        archive.unlink()

    print(f'\nDataset ready: {training.parent}')
    print(f'Train with: bash training_scripts/run_pipeline.sh --huiyu-dir {training.parent}')


if __name__ == '__main__':
    main()
