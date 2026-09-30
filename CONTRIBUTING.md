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

## Releasing

1. Raise `versionCode` and `versionName` in `app/build.gradle`, `version` in `CITATION.cff` and `pyproject.toml`, and add a section to `CHANGELOG.md`.
2. Tag and push: `git tag -a v1.2.3 -m "ASD Screening 1.2.3" && git push origin v1.2.3`. The Release workflow tests the app, builds one APK per CPU family and publishes them with the changelog section as release notes.

The workflow signs the APKs with the key stored in four repository secrets. Without them it falls back to a throwaway debug key, and every release then needs a clean install. A repository admin creates the key once:

```bash
keytool -genkeypair -keystore release.jks -storetype PKCS12 -keyalg RSA -keysize 4096 \
  -validity 10950 -alias asd-screening -dname "CN=ASD Screening, O=University of Catania, C=IT"
base64 -i release.jks | gh secret set ASD_KEYSTORE_BASE64
gh secret set ASD_KEYSTORE_PASSWORD
gh secret set ASD_KEY_ALIAS --body asd-screening
gh secret set ASD_KEY_PASSWORD   # same as the keystore password for PKCS12
```

Keep `release.jks` and its password somewhere safe and outside the repository. Android refuses to update an app signed with a different key.

Connecting the repository to [Zenodo](https://zenodo.org/account/settings/github/) archives every release with a DOI; `.zenodo.json` already holds the metadata.
