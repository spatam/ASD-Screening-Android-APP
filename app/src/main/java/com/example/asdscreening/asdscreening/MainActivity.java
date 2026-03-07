package com.example.asdscreening;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AlertDialog;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.android.material.progressindicator.LinearProgressIndicator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import android.util.Log;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private static final int TOTAL_IMAGES = 300;
    private static final long STIMULUS_MS = 3000;
    private static final long GRAY_MS = 1000;

    private ImageView stimulusImage;
    private PreviewView cameraPreview;
    private FaceOverlayView faceOverlay;
    private LinearProgressIndicator progress;
    private TextView statusText;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private CameraGazeTracker gazeTracker;
    private StimulusRepository stimulusRepo;
    private OnnxEnsembleRunner onnxRunner;

    private int currentIndex = 0;
    private boolean running = false;

    private final List<Float> perImageScores = new ArrayList<>();

    private final ActivityResultLauncher<String> requestCameraPermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startPipeline();
                } else {
                    statusText.setText("Camera permission denied. App can run in DEMO gaze mode without camera.");
                    startPipelineDemo();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        showGlassesDialog();

        stimulusImage = findViewById(R.id.stimulusImage);
        cameraPreview = findViewById(R.id.cameraPreview);
        faceOverlay = findViewById(R.id.faceOverlay);
        //cameraPreview.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        progress = findViewById(R.id.progress);
        statusText = findViewById(R.id.statusText);

        progress.setMax(TOTAL_IMAGES);

        stimulusRepo = new StimulusRepository(getAssets());
        try {
            String[] list = getAssets().list("models");
            Log.d("ASSETS", "assets/models = " + Arrays.toString(list));
            onnxRunner = new OnnxEnsembleRunner(this);
            Log.i("ONNX", "ONNX runner READY");
            statusText.setText("ONNX ready");
        } catch (Exception e) {
            onnxRunner = null;
            Log.e("ONNX", "ONNX init FAILED", e);
            statusText.setText("ONNX init FAILED: " + e.toString());
        }

        // Ask permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startPipeline();
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA);
        }
    }

    private void startPipeline() {
        try {
        gazeTracker = new CameraGazeTracker(this, cameraPreview, faceOverlay);
            gazeTracker.start();
            startStimulusLoop();
        } catch (Exception e) {
            statusText.setText("Camera init failed. Falling back to DEMO gaze. " + e.getMessage());
            startPipelineDemo();
        }
    }
    /*
    private void startPipeline() {
        try {
            cameraPreview.post(() -> {
                try {
                    gazeTracker = new CameraGazeTracker(MainActivity.this, cameraPreview);
                    gazeTracker.start();
                    startStimulusLoop();
                } catch (Exception e) {
                    statusText.setText("Camera init failed. Falling back to DEMO gaze. " + e.getMessage());
                    startPipelineDemo();
                }
            });
        } catch (Exception e) {
            statusText.setText("Camera init failed. Falling back to DEMO gaze. " + e.getMessage());
            startPipelineDemo();
        }
    }
    */


    @Override
    protected void onStop() {
        super.onStop();
        running = false;
        mainHandler.removeCallbacksAndMessages(null);

        if (gazeTracker != null) {
            gazeTracker.stop();
            gazeTracker = null;
        }
    }

    private void startPipelineDemo() {
        gazeTracker = CameraGazeTracker.createDemo(this, cameraPreview);
        gazeTracker.start();
        startStimulusLoop();
    }

    private void startStimulusLoop() {
        if (onnxRunner == null) {
            statusText.setText("ONNX runner not ready.");
            return;
        }
        running = true;
        currentIndex = 0;
        perImageScores.clear();
        nextStimulus();
    }

    private void nextStimulus() {
        if (!running) return;

        if (currentIndex >= TOTAL_IMAGES) {
            finalizeAndShow();
            return;
        }

        progress.setProgress(currentIndex);

        // Load stimulus bitmap (224x224 will be done in preprocessing)
        final Bitmap bmp;
        try {
            bmp = stimulusRepo.loadStimulusBitmap(currentIndex);
        } catch (IOException e) {
            statusText.setText("Missing stimulus asset at index " + currentIndex + ". Add assets/stimuli/0001.png ...");
            // Show gray and stop
            stimulusImage.setImageBitmap(makeGrayBitmap());
            running = false;
            return;
        }

        // Show stimulus
        stimulusImage.setImageBitmap(bmp);

        // Start capture window for this stimulus
        long t0 = System.currentTimeMillis();
        gazeTracker.beginStimulusWindow(t0, t0 + STIMULUS_MS);

        mainHandler.postDelayed(() -> {
            // Switch to gray interval
            stimulusImage.setImageBitmap(makeGrayBitmap());

            // End stimulus window, compute fixations
            List<Fixation> fixations = gazeTracker.endStimulusWindowAndGetFixations();

            // Build model inputs and run inference (off main thread)
            new Thread(() -> {
                try {
                    float pAsd = InferencePipeline.runPerImage(
                            bmp,
                            fixations,
                            onnxRunner
                    );

                    synchronized (perImageScores) {
                        perImageScores.add(pAsd);
                    }

                    float runningScore = ScoreAggregator.runningMean(perImageScores);
                    String verdict = ScoreAggregator.verdictText(runningScore);

                    mainHandler.post(() -> {
                        statusText.setText(
                                "Image " + (currentIndex + 1) + "/" + TOTAL_IMAGES +
                                        " | p(ASD)=" + String.format("%.3f", pAsd) +
                                        " | running=" + String.format("%.3f", runningScore) +
                                        " | " + verdict +
                                        "\nFixations used: " + fixations.size()
                        );

                        // Feed overlay with the latest probability (0..1)
                        if (faceOverlay != null) {
                            faceOverlay.setAsdProbability(pAsd);
                        }
                    });

                } catch (Exception ex) {
                    mainHandler.post(() -> statusText.setText("Inference error: " + ex.getMessage()));
                }
            }).start();

            // Wait gray interval, then next stimulus
            mainHandler.postDelayed(() -> {
                currentIndex += 1;
                nextStimulus();
            }, GRAY_MS);

        }, STIMULUS_MS);
    }

    private void finalizeAndShow() {
        progress.setProgress(TOTAL_IMAGES);
        float finalScore = ScoreAggregator.runningMean(perImageScores);
        String verdict = ScoreAggregator.verdictText(finalScore);
        statusText.setText("DONE. Final p(ASD)=" + String.format("%.3f", finalScore) + " | " + verdict);
        stimulusImage.setImageBitmap(makeGrayBitmap());
        running = false;
    }

    private Bitmap makeGrayBitmap() {
        Bitmap b = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
        b.eraseColor(Color.GRAY);
        return b;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (gazeTracker != null) gazeTracker.stop();
        if (onnxRunner != null) onnxRunner.close();
    }


    private void showGlassesDialog() {
        // Simple, non-blocking UX hint.
        new AlertDialog.Builder(this)
                .setTitle("Eye tracking notice")
                .setMessage("For more accurate eye tracking, please remove glasses if possible.\n\nKeep your face well-lit and centered.")
                .setPositiveButton("OK", (d, w) -> d.dismiss())
                .setCancelable(true)
                .show();
    }
}
