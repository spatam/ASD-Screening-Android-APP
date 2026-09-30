# Contributing

Bug reports, questions and pull requests are welcome. Please open an issue first for larger changes.

## Set up

- Android: JDK 17 or newer and the Android SDK (Android Studio installs both). Build with `./gradlew assembleDebug`.
- Python 3.10 or newer: `pip install -e ".[dev]"`. `requirements.txt` pins the exact versions behind the released models.
- Data: `python scripts/fetch_saliency4asd.py` downloads and checks Saliency4ASD.

## Before you open a pull request

```bash
./gradlew testDebugUnitTest lintDebug
ruff check .
pytest
```

CI runs the same commands on every push.

## Rules that protect the results

- The app must feed the models exactly what training fed them. If you change `asd_gaze/preprocessing.py`, change `Preprocessing.java` or `PilResize.java` to match, run `python scripts/make_parity_golden.py` and commit the new `app/src/test/resources/parity_golden.txt`.
- Keep cross-validation grouped by child (`StratifiedGroupKFold` on subject groups). A child in both the training and the validation split of a fold inflates every metric.
- Never commit data, stimuli, checkpoints or run folders. The `.gitignore` covers them. The Saliency4ASD pictures come from MIT1003 and cannot be redistributed.
- New ONNX models need an entry in `models/SHA256SUMS` and in the model card.
