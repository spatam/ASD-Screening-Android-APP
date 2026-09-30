package it.unict.dmi.asdscreening;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.android.material.progressindicator.LinearProgressIndicator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Session flow:
 *   (first run: import the Saliency4ASD stimuli) -> wait for face -> 9-point calibration
 *   -> [stimulus 3 s -> grey 1 s] x N -> re-calibration every RECALIBRATE_EVERY images
 *   -> session score (sigmoid of the mean per-image logit).
 *
 * Images whose gaze window is unreliable (face lost, blinks, no fixation on the picture)
 * are NOT sent to the model: feeding an empty gaze sequence produces the same prediction
 * for every subject, which was one of the causes of identical results.
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private static final int TOTAL_IMAGES = 300;
    private static final long STIMULUS_MS = 3000;
    private static final long GRAY_MS = 1000;

    // calibration
    private static final long CAL_POINT_MS = 1600;       // target on screen
    private static final long CAL_SETTLE_MS = 700;       // ignore saccade + settling
    private static final long CAL_TAIL_MS = 100;         // ignore last ms (anticipation)
    private static final int CAL_MAX_ATTEMPTS = 3;
    private static final float CAL_MAX_ERR_REL = 0.10f;  // mean error / diagonal of area
    private static final int RECALIBRATE_EVERY = 50;     // images (head drifts over 20 min)
    private static final float NOMINAL_STIM_ASPECT = 1280f / 1024f;

    // quality gate per image
    private static final float MIN_VALID_RATIO = 0.5f;
    private static final int MIN_FIXATIONS = 1;

    private ImageView stimulusImage;
    private PreviewView cameraPreview;
    private FaceOverlayView faceOverlay;
    private LinearProgressIndicator progress;
    private TextView statusText;
    private CalibrationView calibView;
    private View brandingCard;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService importExecutor = Executors.newSingleThreadExecutor();

    private CameraGazeTracker gazeTracker;
    private StimulusRepository stimulusRepo;
    private OnnxEnsembleRunner onnxRunner;

    private int currentIndex = 0;
    private int lastCalibratedAt = -1;
    private int skippedImages = 0;
    private boolean running = false;
    private boolean sessionDone = false;
    private boolean pipelineStarted = false;   // false while stimuli are still being imported
    private boolean useDemo = false;
    private Bitmap grayBitmap;

    private final List<Float> perImageScores = Collections.synchronizedList(new ArrayList<>());

    private final ActivityResultLauncher<Uri> pickStimuliFolder =
            registerForActivityResult(new ActivityResultContracts.OpenDocumentTree(), uri -> {
                if (uri == null) showStimuliDialog();
                else importStimuli(uri);
            });

    private final ActivityResultLauncher<String> requestCameraPermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startPipeline();
                } else {
                    statusText.setText("Camera permission denied. Running in DEMO gaze mode (synthetic gaze).");
                    useDemo = true;
                    startPipeline();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        // Android 15+ draws apps edge to edge: keep every view clear of the system bars.
        // Gaze and calibration use on-screen coordinates, so the padding needs no other change.
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(android.R.id.content), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return insets;
        });

        stimulusImage = findViewById(R.id.stimulusImage);
        cameraPreview = findViewById(R.id.cameraPreview);
        faceOverlay = findViewById(R.id.faceOverlay);
        progress = findViewById(R.id.progress);
        statusText = findViewById(R.id.statusText);
        progress.setMax(TOTAL_IMAGES);

        // Branding card (UniCT logos, department, author), centred over the stimulus area.
        // Visible before the test and on the final result; hidden during calibration/stimuli
        // so the logos do not attract fixations.
        brandingCard = getLayoutInflater().inflate(R.layout.view_branding_header, null, false);
        FrameLayout.LayoutParams blp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER);
        int m = Math.round(24 * getResources().getDisplayMetrics().density);
        blp.setMargins(m, m, m, m);
        addContentView(brandingCard, blp);

        // Calibration target layer on top of everything (no layout change needed)
        calibView = new CalibrationView(this);
        addContentView(calibView, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        grayBitmap = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
        grayBitmap.eraseColor(Color.GRAY);
        stimulusImage.setImageBitmap(grayBitmap);

        stimulusRepo = new StimulusRepository(this);
        try {
            String[] list = getAssets().list("models");
            Log.d("ASSETS", "assets/models = " + Arrays.toString(list));
            onnxRunner = new OnnxEnsembleRunner(this);
            statusText.setText("ONNX ready");
        } catch (Exception e) {
            onnxRunner = null;
            Log.e("ONNX", "ONNX init FAILED", e);
            statusText.setText("ONNX init FAILED: " + e);
        }

        if (stimulusRepo.isComplete()) showGlassesDialog(this::requestCameraThenStart);
        else showStimuliDialog();
    }

    private void requestCameraThenStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startPipeline();
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA);
        }
    }

    // ------------------------------------------------------------------------------------
    // Stimuli import (the public APK ships without the Saliency4ASD pictures)
    // ------------------------------------------------------------------------------------

    private void showStimuliDialog() {
        new AlertDialog.Builder(this)
                .setTitle(R.string.stimuli_title)
                .setMessage(R.string.stimuli_message)
                .setCancelable(false)
                .setPositiveButton(R.string.stimuli_choose, (d, w) -> pickStimuliFolder.launch(null))
                .setNeutralButton(R.string.stimuli_open_zenodo, (d, w) -> {
                    try {
                        startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(getString(R.string.stimuli_zenodo_url))));
                    } catch (ActivityNotFoundException e) {
                        Log.w(TAG, "No browser available", e);
                    }
                    mainHandler.postDelayed(this::showStimuliDialog, 500);
                })
                .setNegativeButton(R.string.stimuli_close, (d, w) -> finish())
                .show();
    }

    private void importStimuli(Uri treeUri) {
        statusText.setText(getString(R.string.stimuli_importing, 0, StimulusRepository.COUNT));
        importExecutor.execute(() -> {
            try {
                StimulusImporter.Result r = StimulusImporter.importFrom(this, treeUri, (done, total) ->
                        mainHandler.post(() -> statusText.setText(getString(R.string.stimuli_importing, done, total))));
                mainHandler.post(() -> {
                    if (r.isComplete() && stimulusRepo.isComplete()) {
                        statusText.setText(R.string.stimuli_done);
                        showGlassesDialog(this::requestCameraThenStart);
                    } else {
                        new AlertDialog.Builder(this)
                                .setTitle(R.string.stimuli_incomplete_title)
                                .setMessage(getString(R.string.stimuli_incomplete_message, r.verified, r.missing, r.rejected))
                                .setCancelable(false)
                                .setPositiveButton(R.string.stimuli_retry, (d, w) -> pickStimuliFolder.launch(null))
                                .setNegativeButton(R.string.stimuli_close, (d, w) -> finish())
                                .show();
                    }
                });
            } catch (Exception e) {
                Log.e(TAG, "Stimulus import failed", e);
                mainHandler.post(() -> {
                    statusText.setText(getString(R.string.stimuli_failed, e.getMessage()));
                    showStimuliDialog();
                });
            }
        });
    }

    // ------------------------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------------------------

    private void startPipeline() {
        if (onnxRunner == null) { statusText.setText("ONNX runner not ready."); return; }
        if (sessionDone) return;
        pipelineStarted = true;
        try {
            gazeTracker = useDemo
                    ? CameraGazeTracker.createDemo(this, cameraPreview)
                    : new CameraGazeTracker(this, cameraPreview, faceOverlay);
            gazeTracker.start();
        } catch (Exception e) {
            Log.e(TAG, "Camera init failed", e);
            statusText.setText("Camera init failed, DEMO gaze mode: " + e.getMessage());
            useDemo = true;
            gazeTracker = CameraGazeTracker.createDemo(this, cameraPreview);
            gazeTracker.start();
        }
        running = true;
        if (gazeTracker.isDemo()) nextStimulus();
        else waitForFace(() -> runCalibration(1, this::nextStimulus));
    }

    /** Polls until a trackable face has been visible continuously for 1 s. */
    private void waitForFace(Runnable then) {
        final long[] since = {-1};
        Runnable poll = new Runnable() {
            @Override public void run() {
                if (!running || gazeTracker == null) return;
                long now = SystemClock.elapsedRealtime();
                if (gazeTracker.isFaceVisible()) {
                    if (since[0] < 0) since[0] = now;
                    if (now - since[0] >= 1000) { then.run(); return; }
                    statusText.setText("Face detected - hold still...");
                } else {
                    since[0] = -1;
                    statusText.setText("Place your face in front of the camera, eyes well lit, phone steady (stand recommended).");
                }
                mainHandler.postDelayed(this, 200);
            }
        };
        mainHandler.post(poll);
    }

    // ------------------------------------------------------------------------------------
    // Calibration
    // ------------------------------------------------------------------------------------

    private void runCalibration(int attempt, Runnable onDone) {
        if (!running) return;
        showBranding(false);
        stimulusImage.setImageBitmap(grayBitmap);
        faceOverlay.setStatusLabel("CALIBRATING");
        gazeTracker.resetCalibration();

        RectF area = nominalStimulusRectOnScreen();
        List<float[]> targets = new ArrayList<>();
        float[] fr = {0.05f, 0.5f, 0.95f};
        for (float fy : fr) for (float fx : fr)
            targets.add(new float[]{area.left + fx * area.width(), area.top + fy * area.height()});
        Collections.shuffle(targets);

        calibView.setMessage("Follow the dot with your EYES (keep your head still)");
        showCalibrationPoint(targets, 0, area, attempt, onDone);
    }

    private void showCalibrationPoint(List<float[]> targets, int i, RectF area, int attempt, Runnable onDone) {
        if (!running) return;
        if (i >= targets.size()) {
            finishCalibration(area, attempt, onDone);
            return;
        }
        float[] t = targets.get(i);
        long now = SystemClock.elapsedRealtime();
        calibView.showTarget(t[0], t[1], CAL_SETTLE_MS);
        gazeTracker.beginCalibrationPoint(i, t[0], t[1], now + CAL_SETTLE_MS, now + CAL_POINT_MS - CAL_TAIL_MS);
        statusText.setText(String.format(Locale.US, "Calibration %d/%d (attempt %d)", i + 1, targets.size(), attempt));
        mainHandler.postDelayed(() -> {
            gazeTracker.endCalibrationPoint();
            showCalibrationPoint(targets, i + 1, area, attempt, onDone);
        }, CAL_POINT_MS);
    }

    private void finishCalibration(RectF area, int attempt, Runnable onDone) {
        calibView.hideTarget();
        calibView.setMessage(null);
        GazeCalibrator.FitResult r = gazeTracker.fitCalibration();
        float diag = (float) Math.hypot(area.width(), area.height());
        boolean good = r != null && r.ok && r.meanErrorPx / diag <= CAL_MAX_ERR_REL;
        String info = (r == null) ? "" : (r.ok
                ? String.format(Locale.US, "mean err %.0f px (%.1f%% of diag), %d pts", r.meanErrorPx, 100f * r.meanErrorPx / diag, r.pointsUsed)
                : r.message);
        Log.i(TAG, "Calibration attempt " + attempt + ": " + info);

        if (good || (r != null && r.ok && attempt >= CAL_MAX_ATTEMPTS)) {
            lastCalibratedAt = currentIndex;
            faceOverlay.setStatusLabel(null);
            statusText.setText("Calibration " + (good ? "OK" : "POOR (continuing)") + ": " + info);
            mainHandler.postDelayed(onDone, 800);
        } else if (attempt < CAL_MAX_ATTEMPTS) {
            statusText.setText("Calibration failed (" + info + "). Retrying...");
            mainHandler.postDelayed(() -> waitForFace(() -> runCalibration(attempt + 1, onDone)), 1200);
        } else {
            // not even a usable fit: keep trying once the face is stable again
            statusText.setText("Calibration impossible (" + info + "). Check lighting / remove glasses.");
            mainHandler.postDelayed(() -> waitForFace(() -> runCalibration(1, onDone)), 2500);
        }
    }

    // ------------------------------------------------------------------------------------
    // Stimuli
    // ------------------------------------------------------------------------------------

    private void nextStimulus() {
        if (!running) return;
        if (currentIndex >= TOTAL_IMAGES) { finalizeAndShow(); return; }

        if (!gazeTracker.isDemo() && currentIndex > 0 && currentIndex % RECALIBRATE_EVERY == 0
                && lastCalibratedAt != currentIndex) {
            waitForFace(() -> runCalibration(1, this::nextStimulus));
            return;
        }

        showBranding(false);
        progress.setProgress(currentIndex);
        final int idx = currentIndex;
        final Bitmap bmp;
        try {
            bmp = stimulusRepo.loadStimulusBitmap(idx);
        } catch (IOException e) {
            statusText.setText("Missing stimulus asset at index " + idx + " (assets/stimuli/" + (idx + 1) + ".png)");
            stimulusImage.setImageBitmap(grayBitmap);
            running = false;
            return;
        }
        stimulusImage.setImageBitmap(bmp);

        // Wait for the layout pass so the image matrix / on-screen rect are final.
        stimulusImage.post(() -> {
            if (!running) return;
            RectF rect = displayedImageRectOnScreen();
            long t0 = SystemClock.elapsedRealtime();
            gazeTracker.beginStimulusWindow(t0, t0 + STIMULUS_MS, rect, bmp.getWidth(), bmp.getHeight());
            mainHandler.postDelayed(() -> endStimulus(idx, bmp), STIMULUS_MS);
        });
    }

    private void endStimulus(int idx, Bitmap bmp) {
        if (!running) return;
        stimulusImage.setImageBitmap(grayBitmap);
        CameraGazeTracker.WindowResult wr = gazeTracker.endStimulusWindow();
        final List<Fixation> fixations = wr.fixations;

        if (wr.validRatio < MIN_VALID_RATIO || fixations.size() < MIN_FIXATIONS) {
            skippedImages++;
            statusText.setText(String.format(Locale.US,
                    "Image %d/%d SKIPPED (valid gaze %.0f%%, fixations %d) | skipped so far: %d",
                    idx + 1, TOTAL_IMAGES, 100f * wr.validRatio, fixations.size(), skippedImages));
        } else {
            final float validRatio = wr.validRatio;
            inferenceExecutor.execute(() -> {
                try {
                    float pAsd = InferencePipeline.runPerImage(bmp, fixations, onnxRunner);
                    perImageScores.add(pAsd);
                    List<Float> scores = snapshotScores();
                    float session = ScoreAggregator.sessionScore(scores);
                    String verdict = ScoreAggregator.verdictText(session, scores.size());
                    boolean decided = ScoreAggregator.hasVerdict(scores.size());
                    mainHandler.post(() -> {
                        statusText.setText(String.format(Locale.US,
                                "Image %d/%d | p(ASD)=%.3f | session=%.3f | %s\nFixations: %d | valid gaze %.0f%% | skipped: %d",
                                idx + 1, TOTAL_IMAGES, pAsd, session, verdict,
                                fixations.size(), 100f * validRatio, skippedImages));
                        faceOverlay.setSessionScore(session, decided);
                    });
                } catch (Exception ex) {
                    Log.e(TAG, "Inference error", ex);
                    mainHandler.post(() -> statusText.setText("Inference error: " + ex.getMessage()));
                }
            });
        }

        mainHandler.postDelayed(() -> {
            currentIndex = idx + 1;
            nextStimulus();
        }, GRAY_MS);
    }

    private void finalizeAndShow() {
        running = false;
        sessionDone = true;
        progress.setProgress(TOTAL_IMAGES);
        stimulusImage.setImageBitmap(grayBitmap);
        showBranding(true);
        statusText.setText("Finishing inference...");
        // single-thread executor: this runs after every pending per-image inference
        inferenceExecutor.execute(() -> {
            List<Float> scores = snapshotScores();
            float finalScore = ScoreAggregator.sessionScore(scores);
            String verdict = ScoreAggregator.verdictText(finalScore, scores.size());
            mainHandler.post(() -> {
                if (scores.isEmpty()) {
                    statusText.setText("DONE. No reliable gaze data: no score computed.");
                } else {
                    statusText.setText(String.format(Locale.US,
                            "DONE. Session score %.3f (study threshold %.3f): %s\nImages scored: %d, skipped: %d. Research use only.",
                            finalScore, ScoreAggregator.STUDY_THRESHOLD, verdict, scores.size(), skippedImages));
                    faceOverlay.setSessionScore(finalScore, ScoreAggregator.hasVerdict(scores.size()));
                }
            });
        });
    }

    private void showBranding(boolean show) {
        if (brandingCard != null) brandingCard.setVisibility(show ? View.VISIBLE : View.GONE);
    }

    private List<Float> snapshotScores() {
        synchronized (perImageScores) { return new ArrayList<>(perImageScores); }
    }

    // ------------------------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------------------------

    /** Rectangle, in screen px, where the current drawable is actually drawn. */
    private RectF displayedImageRectOnScreen() {
        int[] loc = new int[2];
        stimulusImage.getLocationOnScreen(loc);
        float cl = loc[0] + stimulusImage.getPaddingLeft();
        float ct = loc[1] + stimulusImage.getPaddingTop();
        float cw = stimulusImage.getWidth() - stimulusImage.getPaddingLeft() - stimulusImage.getPaddingRight();
        float ch = stimulusImage.getHeight() - stimulusImage.getPaddingTop() - stimulusImage.getPaddingBottom();

        Drawable d = stimulusImage.getDrawable();
        if (d == null || d.getIntrinsicWidth() <= 0 || d.getIntrinsicHeight() <= 0
                || stimulusImage.getScaleType() == ImageView.ScaleType.FIT_XY) {
            return new RectF(cl, ct, cl + cw, ct + ch);
        }
        RectF r = new RectF(0, 0, d.getIntrinsicWidth(), d.getIntrinsicHeight());
        stimulusImage.getImageMatrix().mapRect(r);
        r.offset(cl, ct);
        // visible part only (CENTER_CROP etc.)
        if (!r.intersect(cl, ct, cl + cw, ct + ch)) return new RectF(cl, ct, cl + cw, ct + ch);
        return r;
    }

    /** Area where a 1280x1024 stimulus is displayed (fitCenter): used for calibration targets. */
    private RectF nominalStimulusRectOnScreen() {
        int[] loc = new int[2];
        stimulusImage.getLocationOnScreen(loc);
        float cw = stimulusImage.getWidth() - stimulusImage.getPaddingLeft() - stimulusImage.getPaddingRight();
        float ch = stimulusImage.getHeight() - stimulusImage.getPaddingTop() - stimulusImage.getPaddingBottom();
        float w = cw, h = cw / NOMINAL_STIM_ASPECT;
        if (h > ch) { h = ch; w = ch * NOMINAL_STIM_ASPECT; }
        float l = loc[0] + stimulusImage.getPaddingLeft() + (cw - w) / 2f;
        float t = loc[1] + stimulusImage.getPaddingTop() + (ch - h) / 2f;
        return new RectF(l, t, l + w, t + h);
    }

    // ------------------------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------------------------

    @Override
    protected void onStop() {
        super.onStop();
        running = false;
        mainHandler.removeCallbacksAndMessages(null);
        if (calibView != null) { calibView.hideTarget(); calibView.setMessage(null); }
        if (gazeTracker != null) { gazeTracker.stop(); gazeTracker = null; }
    }

    @Override
    protected void onRestart() {
        super.onRestart();
        // Resume from the interrupted image; the subject may have moved: recalibrate.
        if (pipelineStarted && !sessionDone && onnxRunner != null
                && (useDemo || ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED)) {
            lastCalibratedAt = -1;
            startPipeline();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (gazeTracker != null) gazeTracker.stop();
        inferenceExecutor.shutdown();
        importExecutor.shutdown();
        if (onnxRunner != null) onnxRunner.close();
    }

    private void showGlassesDialog(Runnable onOk) {
        new AlertDialog.Builder(this)
                .setTitle("Eye tracking notice")
                .setMessage(getString(R.string.disclaimer) + "\n\nFor accurate eye tracking:\n\n"
                        + "- remove glasses if possible\n"
                        + "- keep your face well lit and centred\n"
                        + "- place the phone on a stand at ~35 cm\n"
                        + "- move only your EYES during the test\n\n"
                        + "A short calibration (follow the dot) will start first.")
                .setPositiveButton("OK", (d, w) -> d.dismiss())
                .setOnDismissListener(d -> onOk.run())
                .setCancelable(true)
                .show();
    }
}
