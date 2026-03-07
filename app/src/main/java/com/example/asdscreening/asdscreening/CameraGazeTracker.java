package com.example.asdscreening;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.Log;
import android.util.Size;

import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;
import com.google.mlkit.vision.facemesh.FaceMeshPoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CameraX + ML Kit Face Mesh — real gaze tracking.
 *
 * COORDINATE SPACE (verified empirically on this device, rotation=270):
 *
 *   InputImage.fromMediaImage(image, rotationDegrees) tells ML Kit to
 *   treat the image as if it were already rotated. ML Kit then returns
 *   landmark coordinates in the POST-ROTATION (upright) frame:
 *     - raw sensor: imgW=640, imgH=480
 *     - rotation=270 → post-rotation frame: frameW=480, frameH=640
 *   So landmarks are in [0..480] x [0..640] space.
 *
 *   PreviewView renders with FILL_CENTER and mirrors X for the front camera.
 *   We replicate that transform with a single Matrix.
 *
 * EYE CENTRE:
 *   ML Kit FACE_MESH returns 468 landmarks (0-467).
 *   Pupil proxy = centroid of 8 eye-ring landmarks (all within 0-467).
 */
public class CameraGazeTracker {

    private static final String TAG = "CameraGazeTracker";

    // Eye-ring landmark indices — all within 0-467, safe for FACE_MESH
    private static final int[] LEFT_EYE_RING  = {33, 133, 159, 145, 160, 144, 153, 158};
    private static final int[] RIGHT_EYE_RING = {263, 362, 386, 374, 385, 373, 380, 387};

    // Extra periocular points for denser blue overlay ring
    private static final int[] LEFT_EYE_EXTRA  = {246, 161, 163,   7, 173, 157, 154, 155};
    private static final int[] RIGHT_EYE_EXTRA = {466, 388, 390, 249, 398, 384, 381, 382};

    private final Context         context;
    private final PreviewView     previewView;
    private final FaceOverlayView overlayView;

    private final ExecutorService       cameraExecutor = Executors.newSingleThreadExecutor();
    private       ProcessCameraProvider cameraProvider;

    private final AtomicBoolean inStimulusWindow = new AtomicBoolean(false);
    private volatile long windowStartMs = 0;
    private volatile long windowEndMs   = 0;

    private final List<GazeSample> windowSamples = new ArrayList<>();

    private final FaceMeshDetector meshDetector; // null in demo mode

    private final AtomicLong lastDetectMs    = new AtomicLong(0L);
    private static final long DETECT_INTERVAL_MS = 120L; // ~8 fps

    /**
     * Interpupillary display scale factor for the green overlay dots.
     * 1.0 = raw centroid position (default).
     * < 1.0 = bring dots closer together  (e.g. 0.85)
     * > 1.0 = push dots further apart     (e.g. 1.15)
     * This affects only the visual overlay, NOT the gaze signal used for fixations.
     */
    private static final float IPD_SCALE = 0.7f;

    /**
     * Scale factor for the blue eye-mesh overlay dots, same logic as IPD_SCALE.
     * Shrinks/expands the mesh ring around each eye independently from the green dots.
     */
    private static final float MESH_SCALE = 0.7f;

    private volatile float   lastGazeX = 0.5f;
    private volatile float   lastGazeY = 0.5f;
    private volatile boolean gazeValid = false;

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    public static CameraGazeTracker createDemo(Context ctx, PreviewView pv) {
        return new CameraGazeTracker(ctx, pv, null, true);
    }

    public CameraGazeTracker(Context ctx, PreviewView pv) {
        this(ctx, pv, null, false);
    }

    public CameraGazeTracker(Context ctx, PreviewView pv, FaceOverlayView ov) {
        this(ctx, pv, ov, false);
    }

    private CameraGazeTracker(Context ctx, PreviewView pv, FaceOverlayView ov, boolean demo) {
        this.context     = ctx;
        this.previewView = pv;
        this.overlayView = ov;

        if (demo) {
            meshDetector = null;
        } else {
            FaceMeshDetectorOptions opts = new FaceMeshDetectorOptions.Builder()
                    .setUseCase(FaceMeshDetectorOptions.FACE_MESH)
                    .build();
            meshDetector = FaceMeshDetection.getClient(opts);
        }
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    public void start() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(context);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Camera provider failed", e);
            }
        }, ContextCompat.getMainExecutor(context));
    }

    public void stop() {
        try { if (cameraProvider != null) cameraProvider.unbindAll(); }
        catch (Exception ignored) {}
        cameraExecutor.shutdownNow();
        if (meshDetector != null) {
            try { meshDetector.close(); } catch (Exception ignored) {}
        }
    }

    // -----------------------------------------------------------------------
    // Stimulus window
    // -----------------------------------------------------------------------

    public void beginStimulusWindow(long startMs, long endMs) {
        windowStartMs = startMs;
        windowEndMs   = endMs;
        synchronized (windowSamples) { windowSamples.clear(); }
        inStimulusWindow.set(true);
    }

    public List<Fixation> endStimulusWindowAndGetFixations() {
        inStimulusWindow.set(false);
        List<GazeSample> copy;
        synchronized (windowSamples) { copy = new ArrayList<>(windowSamples); }
        return FixationDetector.detectFixations(copy);
    }

    // -----------------------------------------------------------------------
    // CameraX binding
    // -----------------------------------------------------------------------

    @SuppressLint("UnsafeOptInUsageError")
    private void bindUseCases() {
        cameraProvider.unbindAll();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        final boolean isDemo = (meshDetector == null);

        analysis.setAnalyzer(cameraExecutor, image -> {
            long now = System.currentTimeMillis();

            if (isDemo) {
                lastGazeX += (float) ((Math.random() - 0.5) * 0.01);
                lastGazeY += (float) ((Math.random() - 0.5) * 0.01);
                lastGazeX = clamp(lastGazeX, 0.05f, 0.95f);
                lastGazeY = clamp(lastGazeY, 0.05f, 0.95f);
                gazeValid = true;
                recordGazeSample(now);
                image.close();
                return;
            }

            if (now - lastDetectMs.get() < DETECT_INTERVAL_MS) {
                if (gazeValid) recordGazeSample(now);
                image.close();
                return;
            }
            lastDetectMs.set(now);

            @SuppressLint("UnsafeOptInUsageError")
            InputImage inputImage;
            try {
                inputImage = InputImage.fromMediaImage(
                        image.getImage(),
                        image.getImageInfo().getRotationDegrees()
                );
            } catch (Exception e) {
                Log.w(TAG, "InputImage failed", e);
                image.close();
                return;
            }

            // ML Kit returns landmarks in the POST-ROTATION frame.
            // With rotation=270 on a 640x480 sensor: frameW=480, frameH=640.
            final int rotation   = image.getImageInfo().getRotationDegrees();
            final boolean swap   = (rotation == 90 || rotation == 270);
            final int frameW     = swap ? image.getHeight() : image.getWidth();
            final int frameH     = swap ? image.getWidth()  : image.getHeight();

            meshDetector.process(inputImage)
                    .addOnSuccessListener(faces -> {
                        if (faces != null && !faces.isEmpty()) {
                            processFaceMesh(faces.get(0), frameW, frameH, now);
                        } else {
                            gazeValid = false;
                            clearOverlay();
                        }
                    })
                    .addOnFailureListener(e -> Log.w(TAG, "FaceMesh failed", e))
                    .addOnCompleteListener(task -> image.close());
        });

        CameraSelector selector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        cameraProvider.bindToLifecycle(
                (LifecycleOwner) context, selector, preview, analysis
        );
    }

    // -----------------------------------------------------------------------
    // Face Mesh processing
    // -----------------------------------------------------------------------

    private void processFaceMesh(FaceMesh face, int frameW, int frameH, long now) {
        List<FaceMeshPoint> pts = face.getAllPoints();
        if (pts == null || pts.size() < 468) {
            Log.d(TAG, "Unexpected mesh size: " + (pts == null ? 0 : pts.size()));
            return;
        }

        int vw = previewView.getWidth();
        int vh = previewView.getHeight();
        if (vw <= 0 || vh <= 0) return;

        // Eye centres: centroid of eye-ring landmarks
        PointF leftCenter  = centroid(pts, LEFT_EYE_RING);
        PointF rightCenter = centroid(pts, RIGHT_EYE_RING);

        // Gaze normalised to [0,1], X mirrored for front camera
        float rawX = (leftCenter.x + rightCenter.x) / 2f / frameW;
        float rawY = (leftCenter.y + rightCenter.y) / 2f / frameH;
        lastGazeX = clamp(1f - rawX, 0f, 1f);
        lastGazeY = clamp(rawY,      0f, 1f);
        gazeValid = true;
        recordGazeSample(now);

        if (overlayView == null) return;

        // Single matrix: frame coords → PreviewView pixel coords
        Matrix toView = buildToViewMatrix(frameW, frameH, vw, vh);

        // Face bounding box
        RectF bb = face.getBoundingBox() != null
                ? new RectF(face.getBoundingBox())
                : estimateFaceRect(pts);
        RectF mappedFace = mapRect(bb, toView);

        // Blue mesh dots, scaled around each eye centre by MESH_SCALE
        List<PointF> meshRaw = new ArrayList<>();
        collectPointsScaled(meshRaw, pts, LEFT_EYE_RING,  leftCenter,  MESH_SCALE);
        collectPointsScaled(meshRaw, pts, RIGHT_EYE_RING, rightCenter, MESH_SCALE);
        collectPointsScaled(meshRaw, pts, LEFT_EYE_EXTRA,  leftCenter,  MESH_SCALE);
        collectPointsScaled(meshRaw, pts, RIGHT_EYE_EXTRA, rightCenter, MESH_SCALE);
        List<PointF> mappedMesh = mapPoints(meshRaw, toView);

        // Green gaze dots (eye centres), optionally scaled by IPD_SCALE
        float midX = (leftCenter.x + rightCenter.x) / 2f;
        float midY = (leftCenter.y + rightCenter.y) / 2f;
        List<PointF> gazeRaw = new ArrayList<>();
        gazeRaw.add(new PointF(
                midX + (leftCenter.x  - midX) * IPD_SCALE,
                midY + (leftCenter.y  - midY) * IPD_SCALE));
        gazeRaw.add(new PointF(
                midX + (rightCenter.x - midX) * IPD_SCALE,
                midY + (rightCenter.y - midY) * IPD_SCALE));
        List<PointF> mappedGaze = mapPoints(gazeRaw, toView);

        FaceOverlayView.OverlayData od =
                new FaceOverlayView.OverlayData(mappedFace, mappedMesh, mappedGaze);
        overlayView.post(() -> overlayView.setData(od));
    }

    // -----------------------------------------------------------------------
    // Coordinate transform
    // -----------------------------------------------------------------------

    /**
     * Builds a Matrix mapping landmark coordinates (post-rotation frame space,
     * frameW × frameH) → PreviewView pixel space.
     *
     * PreviewView uses FILL_CENTER and mirrors X for the front camera:
     *
     *   scale = max(viewW/frameW, viewH/frameH)
     *   dx    = (viewW − frameW·scale) / 2
     *   dy    = (viewH − frameH·scale) / 2
     *   mirrorX around (viewW/2, viewH/2)
     *
     * All operations are chained with Matrix.post* (each appends AFTER current).
     */
    private static Matrix buildToViewMatrix(int frameW, int frameH, int viewW, int viewH) {
        float scale = Math.max((float) viewW / frameW, (float) viewH / frameH);
        float dx    = (viewW - frameW * scale) / 2f;
        float dy    = (viewH - frameH * scale) / 2f;

        Matrix m = new Matrix();
        m.setScale(scale, scale);
        m.postTranslate(dx, dy);
        // Mirror X for front camera: pivot at horizontal centre of view
        m.postScale(-1f, 1f, viewW / 2f, viewH / 2f);
        return m;
    }

    private static List<PointF> mapPoints(List<PointF> src, Matrix m) {
        if (src == null || src.isEmpty()) return Collections.emptyList();
        float[] coords = new float[src.size() * 2];
        for (int i = 0; i < src.size(); i++) {
            coords[i * 2]     = src.get(i).x;
            coords[i * 2 + 1] = src.get(i).y;
        }
        m.mapPoints(coords);
        List<PointF> out = new ArrayList<>(src.size());
        for (int i = 0; i < src.size(); i++) {
            out.add(new PointF(coords[i * 2], coords[i * 2 + 1]));
        }
        return out;
    }

    private static RectF mapRect(RectF src, Matrix m) {
        RectF dst = new RectF(src);
        m.mapRect(dst);
        return dst;
    }

    // -----------------------------------------------------------------------
    // Geometry helpers
    // -----------------------------------------------------------------------

    private static PointF centroid(List<FaceMeshPoint> pts, int[] indices) {
        float sx = 0f, sy = 0f;
        for (int idx : indices) {
            sx += pts.get(idx).getPosition().getX();
            sy += pts.get(idx).getPosition().getY();
        }
        return new PointF(sx / indices.length, sy / indices.length);
    }

    private static void collectPoints(List<PointF> dst, List<FaceMeshPoint> pts, int[] indices) {
        for (int idx : indices) {
            dst.add(new PointF(
                    pts.get(idx).getPosition().getX(),
                    pts.get(idx).getPosition().getY()
            ));
        }
    }

    /** Like collectPoints but scales each point around a pivot by the given factor. */
    private static void collectPointsScaled(List<PointF> dst, List<FaceMeshPoint> pts,
                                            int[] indices, PointF pivot, float scale) {
        for (int idx : indices) {
            float x = pts.get(idx).getPosition().getX();
            float y = pts.get(idx).getPosition().getY();
            dst.add(new PointF(
                    pivot.x + (x - pivot.x) * scale,
                    pivot.y + (y - pivot.y) * scale
            ));
        }
    }

    private static RectF estimateFaceRect(List<FaceMeshPoint> pts) {
        float minX = Float.MAX_VALUE, maxX = -Float.MAX_VALUE;
        float minY = Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (FaceMeshPoint p : pts) {
            float x = p.getPosition().getX();
            float y = p.getPosition().getY();
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
        return new RectF(minX, minY, maxX, maxY);
    }

    // -----------------------------------------------------------------------
    // Gaze recording
    // -----------------------------------------------------------------------

    private void recordGazeSample(long now) {
        if (inStimulusWindow.get() && now >= windowStartMs && now <= windowEndMs) {
            synchronized (windowSamples) {
                windowSamples.add(new GazeSample(now, lastGazeX, lastGazeY, true));
            }
        }
    }

    private void clearOverlay() {
        if (overlayView != null) {
            overlayView.post(() -> overlayView.setData(
                    new FaceOverlayView.OverlayData(
                            null,
                            Collections.emptyList(),
                            Collections.emptyList()
                    )
            ));
        }
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // -----------------------------------------------------------------------
    // Public data class
    // -----------------------------------------------------------------------

    public static class GazeSample {
        public final long    tMs;
        public final float   xNorm;
        public final float   yNorm;
        public final boolean valid;

        public GazeSample(long tMs, float xNorm, float yNorm, boolean valid) {
            this.tMs   = tMs;
            this.xNorm = xNorm;
            this.yNorm = yNorm;
            this.valid = valid;
        }
    }
}
