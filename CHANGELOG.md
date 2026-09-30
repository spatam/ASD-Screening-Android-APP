# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org).

## [1.0.1] - 2026-09-30

### App

- The APKs are signed with the project's release key. Its certificate has the SHA-256 digest `45d9fac7aef32ba847982ff331ff6cad5f3b259ecd32b4d83fc3d129ba986039`, and every release lists it. Version 1.0.0 carried a temporary key that Android will not update, so uninstall 1.0.0 once before installing 1.0.1. Later versions update in place.

### Documentation

- README artwork drawn with text outlines, so the labels keep their layout in every browser.
- University of Catania and IPLab logos in the README footer.
- Zenodo metadata (`.zenodo.json`) and a release guide in `CONTRIBUTING.md`.

## [1.0.0] - 2026-09-30

First public release, published with the paper in the IEEE Open Journal of the Computer Society ([10.1109/OJCS.2026.3738736](https://doi.org/10.1109/OJCS.2026.3738736)).

### App

- The app now runs the **deployment** students (`subject_all` protocol), the setting the paper evaluates for unseen children on a fixed set of pictures. The retained-subject students of the headline table moved to `models/retained_subject/`.
- **Preprocessing now matches training.** Three differences between the Java code and the Python pipeline changed the model inputs:
  - Stimuli were resized with `Bitmap.createScaledBitmap`. `PilResize` now reproduces Pillow's bilinear filter bit for bit.
  - Fixations were placed on the heatmap as if every picture filled 1280×1024. They now use stimulus pixels, as in the Saliency4ASD files.
  - With more than 25 fixations, the gaze sequence was min-max scaled over the first 25 only. It now uses all of them, like the training code.

  On 600 Saliency4ASD records the first two differences together moved p(ASD) by 0.11 on average. Unit tests now compare the Java code with values produced by the Python code.
- The session score is the sigmoid of the mean per-picture logit, the aggregation behind the paper's per-subject AUC of 0.959. The verdict appears after 40 pictures and compares the score with the study threshold (0.492). The old fixed cut-offs (0.85 and 0.15) were never reached by any of the 28 dataset children.
- The APK ships without stimuli, because the MIT1003 photos cannot be redistributed. On first launch the app imports them from the Saliency4ASD archive and checks each file against its SHA-256.
- No network access: the manifest removes the INTERNET and ACCESS_NETWORK_STATE permissions that ML Kit adds, and backups are off.
- A research-use notice opens every session. The app handles Android 15+ edge-to-edge insets and falls back to input names by tensor rank.
- Package renamed to `it.unict.dmi.asdscreening`. One APK per CPU family (about 65 MB each).

### Build and code

- The project builds from a fresh clone: Gradle wrapper 9.8, Android Gradle Plugin 9.4, compile and target SDK 36, minimum SDK 26. A stale copy of older sources that broke compilation is gone.
- The Python training package `asd_gaze` is included, with the three protocols (`subject`, `stimulus`, `subject_all`), the analysis scripts behind the paper's tables, `pyproject.toml` and pinned `requirements.txt`.
- `compute_full_metrics` reports per-subject AUC in logit space by default, matching the paper (0.9592 on the deployment run).
- The ONNX exporter writes the input names the app expects (`gaze_seq`, `stimulus_image`).
- The random-heatmap control uses a stable CRC32 seed. It used Python's salted `hash`, so its surrogate fixations changed between processes.
- New: `scripts/fetch_saliency4asd.py` (download, MD5 and SHA-256 checks, extraction), `examples/infer_onnx.py`, a model card with checksums, CI for Android and Python, and a release workflow.

### License

- Code and models are released under Apache-2.0 (see `NOTICE`). Earlier commits stay under CC0 1.0.
