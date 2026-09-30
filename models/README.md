# Model card

Two sets of five ONNX students (MobileNetV3-Small spatial stream plus a one-layer temporal transformer, 1,602,625 parameters each). Each file is one fold of a 5-fold `StratifiedGroupKFold` (`shuffle=True`, `random_state=42`), and the prediction for a picture is the mean of the five logits.

| Set | Location | Protocol | Records | Student OOF AUC | Per-subject AUC | Used by |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Deployment | [`app/src/main/assets/models/`](../app/src/main/assets/models) | `subject_all`: all 600 scanpath files, at most 14 subject slots per file, ASD and TD balanced within every picture, folds grouped by child | 6770 | 0.6859 | 0.9592 | Android app |
| Retained subject | [`retained_subject/`](retained_subject) | `subject`: only files with exactly 14 subjects, folds grouped by child | 2216 | 0.9894 | 1.0000 | paper headline table |

The two sets answer different questions. In the retained-subject subset 134 of the 149 pictures appear in one class only, so the identity of the picture alone reaches an AUC of 0.9825 and the headline score measures little gaze signal. The deployment protocol balances ASD and TD viewers within every picture, which makes picture identity worth an AUC of exactly 0.5000. Its per-image AUC of 0.6859 therefore comes from gaze, and pooling a session's logits lifts it to 0.9592 per child. The app shows the same fixed pictures to children it has never seen, so it ships the deployment set.

## Inputs and output

| Name | Shape | Content |
| --- | --- | --- |
| `gaze_seq` | `[batch, seq_len, 3]`, `seq_len` = 25 | per fixation: x and y min-max scaled over the record, `log1p(duration_ms) / 10`, zero-padded |
| `stimulus_image` | `[batch, 3, 224, 224]` | stimulus resized with Pillow bilinear, blended `0.55 * image + 0.45 * jet(heatmap)`, ImageNet-normalised |
| `logit` | `[batch]` (deployment), `[1]` (retained) | log-odds of ASD |

The heatmap places each fixation, given in stimulus pixels, at `x / 1280 * 223` and `y / 1024 * 223`, weights it by `log1p(duration_ms)`, blurs with a Gaussian of sigma 10 and scales the peak to 1. [`asd_gaze/preprocessing.py`](../asd_gaze/preprocessing.py) is the reference implementation, and the app's Java port is tested against it.

## Session score

The app turns the per-picture logits of a session into one score: the sigmoid of their mean. At the end of the session it compares the score with 0.4922, the Youden cut of the 28 Saliency4ASD children under the deployment protocol, and it reports a side only after 40 pictures. The paper measures a per-subject AUC of 0.947 for 40 pictures (under three minutes) and 0.959 for the full set. With the threshold and a calibration fitted leave-one-subject-out, the paper reports accuracy 0.821, sensitivity 0.857 and specificity 0.786.

## Training data and procedure

- Data: Saliency4ASD (Duan et al., MMSys 2019, [10.1145/3304109.3325818](https://doi.org/10.1145/3304109.3325818)), 14 children with ASD aged 5 to 12 and 14 typically developing children, 300 pictures shown for 3 s each on a Tobii T120.
- Teacher: frozen ViT-B/16 (ImageNet-21k weights from timm) plus a 3-layer temporal transformer, 154K trainable parameters.
- Student: MobileNetV3-Small (ImageNet weights from timm) distilled from the teacher with a KL plus BCE loss (temperature 4, alpha 0.7). Under the deployment protocol the student learns from the Phase 1 teacher, because the Phase 2 augmentation lowered the teacher's AUC there.
- Export: PyTorch 2.10, opset 18, legacy TorchScript exporter. ONNX Runtime matches the PyTorch checkpoint within 6e-7 on the deployment set.

## Intended use and limits

- Research on gaze-based screening and on-device distillation. The models are not a medical device and must not be used to diagnose or rule out autism.
- 28 children from one laboratory and one eye tracker. A confidence interval of about ±0.10 surrounds the per-subject AUC, and nothing is known about other ages, cultures, devices or clinical populations.
- Phone front cameras are far less precise than the Tobii eye tracker used to collect the training data. The app's gaze estimates carry that error into the model inputs.
- The deployment models saw all 28 dataset children during training: each child sits in the training split of four of the five folds. Scores computed on those children are optimistic.

## Checksums

[`SHA256SUMS`](SHA256SUMS) lists the SHA-256 of every file. Check them with:

```bash
shasum -a 256 -c models/SHA256SUMS
```

## License

Apache-2.0, like the code. Cite the Saliency4ASD paper when you use the models, as its CC BY 4.0 license requires.
