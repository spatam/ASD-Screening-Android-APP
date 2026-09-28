# ASDScreening (Android / Java)

This is a complete Android Studio project (Java) implementing your ASD vs TD screening pipeline:

- 300 stimulus images (3s each) + 1s gray interval
- CameraX front camera running (for future gaze estimation)
- Fixation extraction (simple dispersion-based; replaceable)
- Per-image preprocessing EXACTLY as provided:
  - gaze_seq (1,25,3)
  - stimulus_image (1,3,224,224) = blended + ImageNet normalized
- ONNX Runtime Mobile inference with 5-fold ensemble
- Running average score displayed live

![app_blur](https://github.com/user-attachments/assets/image_app)

## Assets you must add

1) ONNX models:

   Place your files here:
   - app/src/main/assets/models/student_fold1.onnx
   - app/src/main/assets/models/student_fold2.onnx
   - app/src/main/assets/models/student_fold3.onnx
   - app/src/main/assets/models/student_fold4.onnx
   - app/src/main/assets/models/student_fold5.onnx

2) Stimuli images (PNG):
   Place 300 images here:
   - app/src/main/assets/stimuli/0001.png ... 0300.png
   - download from https://zenodo.org/records/2647418

## Notes

- Current gaze estimation is DEMO (center-of-screen jitter).
  Replace `CameraGazeTracker.DemoGazeEstimator` with a real estimator (MediaPipe Face Mesh / ML Kit + calibration).
- Fixations are detected from raw gaze samples. If you already have fixations from your algorithm,
  bypass `FixationDetector` and feed fixations directly to `InferencePipeline`.

## Build

- Open in Android Studio
- Sync Gradle
- Run on device (recommended arm64)

