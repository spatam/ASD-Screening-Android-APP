package com.example.asdscreening;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;

import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;
import com.google.mlkit.vision.facemesh.FaceMeshPoint;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * CameraX + ML Kit Face Mesh + iris localisation + per-subject calibration.
 *
 * Pipeline per camera frame (camera thread):
 *   ML Kit Face Mesh (eyelid contours, eye corners, nose)
 *     -> PupilLocalizer on the Y plane (iris centre inside each eye, normalised by eye width)
 *     -> feature f = [ex, ey, headYaw, headPitch]
 *     -> calibration mode : stored with the current on-screen target
 *        stimulus mode    : GazeCalibrator -> screen px -> normalised to the displayed
 *                           stimulus rectangle -> One-Euro filter -> GazeSample
 *     -> overlay (eyelid contours + pupils) mapped into the PreviewView coordinates.
 *
 * All timestamps use SystemClock.elapsedRealtime().
 */
public class CameraGazeTracker {

    private static final String TAG = "CameraGazeTracker";

    // ---- MediaPipe/ML Kit canonical face-mesh indices --------------------------------------
    // Eyelid contours: image-left corner -> upper lid -> image-right corner -> lower lid back.
    private static final int[] RIGHT_EYE_CONTOUR = {33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7};
    private static final int[] LEFT_EYE_CONTOUR  = {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382};
    private static final int R_CORNER_A = 33, R_CORNER_B = 133;
    private static final int L_CORNER_A = 362, L_CORNER_B = 263;
    private static final int NOSE_TIP = 1;

    /** Analysis resolution in the portrait (upright) orientation. Higher = more iris pixels. */
    private static final Size ANALYSIS_TARGET = new Size(960, 1280);

    /** A face is "visible" if a valid measurement arrived within this time. */
    private static final long FACE_TIMEOUT_MS = 400;

    // ---- result of a stimulus window -------------------------------------------------------
    public static class WindowResult {
        public final List<Fixation> fixations;
        public final int totalSamples;
        public final int validSamples;
        public final float validRatio;
        public final List<GazeSample> samples;

        WindowResult(List<Fixation> f, List<GazeSample> s) {
            fixations = f;
            samples = s;
            totalSamples = s.size();
            int v = 0;
            for (GazeSample g : s) if (g.valid) v++;
            validSamples = v;
            validRatio = s.isEmpty() ? 0f : (float) v / s.size();
        }
    }

    // ---- collaborators ----------------------------------------------------------------------
    private final Context context;
    private final PreviewView previewView;
    private final FaceOverlayView overlayView;
    private final boolean demo;

    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    /** Listener executor that never throws after stop() (late ML Kit callbacks still close the frame). */
    private final Executor listenerExecutor = r -> {
        try { cameraExecutor.execute(r); } catch (RejectedExecutionException e) { r.run(); }
    };
    private ScheduledExecutorService demoExecutor;
    private ProcessCameraProvider cameraProvider;
    private final FaceMeshDetector meshDetector;
    private volatile boolean stopped = false;

    // camera-thread-only state
    private final PupilLocalizer leftLoc = new PupilLocalizer();
    private final PupilLocalizer rightLoc = new PupilLocalizer();
    private final PupilLocalizer.Result leftRes = new PupilLocalizer.Result();
    private final PupilLocalizer.Result rightRes = new PupilLocalizer.Result();
    private byte[] luma = new byte[0];

    // view geometry, refreshed on the UI thread
    private static final class ViewGeom {
        final int pw, ph; final float offX, offY;
        ViewGeom(int pw, int ph, float offX, float offY) { this.pw = pw; this.ph = ph; this.offX = offX; this.offY = offY; }
    }
    private volatile ViewGeom viewGeom;

    // calibration
    private final GazeCalibrator calibrator = new GazeCalibrator();
    private final Object calLock = new Object();
    private int calPointId = -1;
    private float calTx, calTy;
    private long calAcceptFrom, calAcceptTo;

    // stimulus window
    private final Object winLock = new Object();
    private boolean inWindow = false;
    private long winStart, winEnd;
    private RectF winRect;                         // displayed stimulus, screen px
    private final List<GazeSample> winSamples = new ArrayList<>();
    private final OneEuroFilter fx = new OneEuroFilter(1.5, 0.5, 1.0);
    private final OneEuroFilter fy = new OneEuroFilter(1.5, 0.5, 1.0);
    private boolean filterPrimed = false;

    private volatile long lastValidMs = 0;
    private volatile float lastOpenness = 0;

    // demo state
    private final Random rnd = new Random();
    private float demoX = 0.5f, demoY = 0.5f, demoTx = 0.5f, demoTy = 0.5f;
    private long demoNextSaccade = 0;

    // ---------------------------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------------------------

    public static CameraGazeTracker createDemo(Context ctx, PreviewView pv) {
        return new CameraGazeTracker(ctx, pv, null, true);
    }

    public CameraGazeTracker(Context ctx, PreviewView pv, FaceOverlayView ov) {
        this(ctx, pv, ov, false);
    }

    private CameraGazeTracker(Context ctx, PreviewView pv, FaceOverlayView ov, boolean demo) {
        this.context = ctx;
        this.previewView = pv;
        this.overlayView = ov;
        this.demo = demo;
        if (demo) {
            meshDetector = null;
        } else {
            FaceMeshDetectorOptions opts = new FaceMeshDetectorOptions.Builder()
                    .setUseCase(FaceMeshDetectorOptions.FACE_MESH)
                    .build();
            meshDetector = FaceMeshDetection.getClient(opts);
        }
    }

    public boolean isDemo() { return demo; }

    // ---------------------------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------------------------

    public void start() {
        stopped = false;
        if (demo) {
            demoExecutor = Executors.newSingleThreadScheduledExecutor();
            demoExecutor.scheduleAtFixedRate(this::demoTick, 0, 33, TimeUnit.MILLISECONDS);
            return;
        }
        View.OnLayoutChangeListener l = (v, a, b, c, d, e, f, g, h) -> refreshViewGeometry();
        previewView.addOnLayoutChangeListener(l);
        if (overlayView != null) overlayView.addOnLayoutChangeListener(l);

        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(context);
        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                // ViewPort is only available once the PreviewView has been laid out
                previewView.post(this::bindUseCases);
            } catch (Exception e) {
                Log.e(TAG, "Camera provider failed", e);
            }
        }, ContextCompat.getMainExecutor(context));
    }

    public void stop() {
        stopped = true;
        try { if (cameraProvider != null) cameraProvider.unbindAll(); } catch (Exception ignored) {}
        if (demoExecutor != null) demoExecutor.shutdownNow();
        cameraExecutor.shutdown();
        if (meshDetector != null) {
            try { meshDetector.close(); } catch (Exception ignored) {}
        }
    }

    /** True if a face with open, trackable eyes has been seen very recently. */
    public boolean isFaceVisible() {
        return demo || SystemClock.elapsedRealtime() - lastValidMs < FACE_TIMEOUT_MS;
    }

    // ---------------------------------------------------------------------------------------
    // Calibration API (UI thread)
    // ---------------------------------------------------------------------------------------

    public void resetCalibration() {
        calibrator.clearSamples();
    }

    /**
     * Starts collecting samples for a target drawn at (screenX, screenY).
     * Samples before acceptFromMs (saccade latency + settling) are ignored.
     */
    public void beginCalibrationPoint(int pointId, float screenX, float screenY,
                                      long acceptFromMs, long acceptToMs) {
        synchronized (calLock) {
            calPointId = pointId;
            calTx = screenX; calTy = screenY;
            calAcceptFrom = acceptFromMs; calAcceptTo = acceptToMs;
        }
    }

    public void endCalibrationPoint() {
        synchronized (calLock) { calPointId = -1; }
    }

    public GazeCalibrator.FitResult fitCalibration() {
        if (demo) return null;
        return calibrator.fit();
    }

    public boolean isCalibrated() {
        return demo || calibrator.isFitted();
    }

    // ---------------------------------------------------------------------------------------
    // Stimulus window API (UI thread)
    // ---------------------------------------------------------------------------------------

    /**
     * @param stimulusRectOnScreen rectangle (screen px) where the stimulus bitmap is actually
     *                             drawn - gaze is normalised to it, as in the training data.
     */
    public void beginStimulusWindow(long startMs, long endMs, RectF stimulusRectOnScreen) {
        synchronized (winLock) {
            winStart = startMs;
            winEnd = endMs;
            winRect = new RectF(stimulusRectOnScreen);
            winSamples.clear();
            fx.reset(); fy.reset(); filterPrimed = false;
            inWindow = true;
        }
    }

    public WindowResult endStimulusWindow() {
        List<GazeSample> copy;
        synchronized (winLock) {
            inWindow = false;
            copy = new ArrayList<>(winSamples);
        }
        return new WindowResult(FixationDetector.detectFixations(copy), copy);
    }

    // ---------------------------------------------------------------------------------------
    // CameraX
    // ---------------------------------------------------------------------------------------

    @SuppressWarnings("deprecation")
    private void bindUseCases() {
        if (stopped || cameraProvider == null) return;
        cameraProvider.unbindAll();
        refreshViewGeometry();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setTargetResolution(ANALYSIS_TARGET)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build();
        analysis.setAnalyzer(cameraExecutor, this::analyze);

        CameraSelector selector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        // Same ViewPort for Preview and ImageAnalysis: the analysis crop rect then covers
        // exactly the field of view shown in the PreviewView -> overlay aligns 1:1.
        UseCaseGroup.Builder group = new UseCaseGroup.Builder()
                .addUseCase(preview)
                .addUseCase(analysis);
        ViewPort vp = previewView.getViewPort();
        if (vp != null) group.setViewPort(vp);
        else Log.w(TAG, "PreviewView not laid out: binding without ViewPort (FILL_CENTER fallback)");

        cameraProvider.bindToLifecycle((LifecycleOwner) context, selector, group.build());
    }

    /** UI thread: cache preview size and preview->overlay offset. */
    private void refreshViewGeometry() {
        int pw = previewView.getWidth(), ph = previewView.getHeight();
        if (pw <= 0 || ph <= 0) return;
        float offX = 0, offY = 0;
        if (overlayView != null) {
            int[] a = new int[2], b = new int[2];
            previewView.getLocationOnScreen(a);
            overlayView.getLocationOnScreen(b);
            offX = a[0] - b[0];
            offY = a[1] - b[1];
        }
        viewGeom = new ViewGeom(pw, ph, offX, offY);
    }

    @SuppressLint("UnsafeOptInUsageError")
    private void analyze(ImageProxy image) {
        final long now = SystemClock.elapsedRealtime();
        if (stopped) { image.close(); return; }

        Image media = image.getImage();
        if (media == null) { image.close(); return; }

        final int rotation = image.getImageInfo().getRotationDegrees();
        final int rawW = image.getWidth(), rawH = image.getHeight();

        // Copy the Y plane (the buffer is reused: the next frame is delivered only after close()).
        ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
        ByteBuffer yb = yPlane.getBuffer();
        yb.rewind();
        int len = yb.remaining();
        if (luma.length < len) luma = new byte[len];
        yb.get(luma, 0, len);
        final int rowStride = yPlane.getRowStride();
        final int pixelStride = yPlane.getPixelStride();
        final Rect crop = new Rect(image.getCropRect());

        InputImage input;
        try {
            input = InputImage.fromMediaImage(media, rotation);
        } catch (Exception e) {
            Log.w(TAG, "InputImage failed", e);
            image.close();
            return;
        }

        meshDetector.process(input)
                .addOnSuccessListener(listenerExecutor, faces -> {
                    try {
                        if (faces != null && !faces.isEmpty()) {
                            processFace(largest(faces), now, rawW, rawH, rotation, rowStride, pixelStride, crop);
                        } else {
                            onInvalid(now);
                            postOverlay(null);
                        }
                    } catch (Throwable t) {
                        Log.w(TAG, "Frame processing failed", t);
                        onInvalid(now);
                    } finally {
                        image.close();
                    }
                })
                .addOnFailureListener(listenerExecutor, e -> {
                    Log.w(TAG, "FaceMesh failed", e);
                    onInvalid(now);
                    image.close();
                })
                .addOnCanceledListener(listenerExecutor, image::close);
    }

    private static FaceMesh largest(List<FaceMesh> faces) {
        FaceMesh best = faces.get(0);
        for (FaceMesh f : faces) {
            if (f.getBoundingBox().width() * f.getBoundingBox().height()
                    > best.getBoundingBox().width() * best.getBoundingBox().height()) best = f;
        }
        return best;
    }

    // ---------------------------------------------------------------------------------------
    // Per-frame processing (camera thread)
    // ---------------------------------------------------------------------------------------

    private void processFace(FaceMesh face, long now, int rawW, int rawH, int rotation,
                             int rowStride, int pixelStride, Rect cropRaw) {
        List<FaceMeshPoint> pts = face.getAllPoints();
        if (pts == null || pts.size() < 468) { onInvalid(now); return; }

        float[] rCont = contour(pts, RIGHT_EYE_CONTOUR);
        float[] lCont = contour(pts, LEFT_EYE_CONTOUR);
        PointF rA = pos(pts, R_CORNER_A), rB = pos(pts, R_CORNER_B);
        PointF lA = pos(pts, L_CORNER_A), lB = pos(pts, L_CORNER_B);
        // guarantee A = image-left corner (robust to mirrored inputs)
        if (rA.x > rB.x) { PointF t = rA; rA = rB; rB = t; }
        if (lA.x > lB.x) { PointF t = lA; lA = lB; lB = t; }

        rightLoc.locate(luma, rowStride, pixelStride, rawW, rawH, rotation, rA.x, rA.y, rB.x, rB.y, rCont, rightRes);
        leftLoc.locate(luma, rowStride, pixelStride, rawW, rawH, rotation, lA.x, lA.y, lB.x, lB.y, lCont, leftRes);

        boolean valid = rightRes.valid && leftRes.valid;
        lastOpenness = 0.5f * (rightRes.openness + leftRes.openness);

        if (valid) {
            // head pose proxies, expressed along/perpendicular to the interocular axis
            float rcx = (rA.x + rB.x) / 2f, rcy = (rA.y + rB.y) / 2f;
            float lcx = (lA.x + lB.x) / 2f, lcy = (lA.y + lB.y) / 2f;
            float ax = lcx - rcx, ay = lcy - rcy;
            float iod = (float) Math.hypot(ax, ay);
            PointF nose = pos(pts, NOSE_TIP);
            float mx = (rcx + lcx) / 2f, my = (rcy + lcy) / 2f;
            float nx = nose.x - mx, ny = nose.y - my;
            float yaw = (nx * ax + ny * ay) / (iod * iod);
            float pitch = (-nx * ay + ny * ax) / (iod * iod);

            float[] f = new float[]{
                    0.5f * (rightRes.ex + leftRes.ex),
                    0.5f * (rightRes.ey + leftRes.ey),
                    yaw, pitch
            };
            lastValidMs = now;
            onFeatures(f, now);
        } else {
            onInvalid(now);
        }

        if (overlayView != null) buildOverlay(face, pts, rCont, lCont, rawW, rawH, rotation, cropRaw, valid);
    }

    private void onFeatures(float[] f, long now) {
        // calibration
        synchronized (calLock) {
            if (calPointId >= 0 && now >= calAcceptFrom && now <= calAcceptTo) {
                calibrator.addSample(calPointId, calTx, calTy, f);
            }
        }
        // stimulus window
        synchronized (winLock) {
            if (!inWindow || now < winStart || now > winEnd) return;
            float[] scr = calibrator.predict(f);
            if (scr == null || winRect == null || winRect.width() <= 0 || winRect.height() <= 0) {
                winSamples.add(new GazeSample(now, 0, 0, false));
                return;
            }
            float xn = (scr[0] - winRect.left) / winRect.width();
            float yn = (scr[1] - winRect.top) / winRect.height();
            // keep wildly off-screen predictions from dragging the filter
            xn = Math.max(-0.5f, Math.min(1.5f, xn));
            yn = Math.max(-0.5f, Math.min(1.5f, yn));
            float sx = (float) fx.filter(xn, now);
            float sy = (float) fy.filter(yn, now);
            filterPrimed = true;
            winSamples.add(new GazeSample(now, sx, sy, true));
        }
    }

    private void onInvalid(long now) {
        synchronized (winLock) {
            if (inWindow && now >= winStart && now <= winEnd) {
                winSamples.add(new GazeSample(now, 0, 0, false));
                if (filterPrimed) { fx.reset(); fy.reset(); filterPrimed = false; }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Overlay
    // ---------------------------------------------------------------------------------------

    private void buildOverlay(FaceMesh face, List<FaceMeshPoint> pts, float[] rCont, float[] lCont,
                              int rawW, int rawH, int rotation, Rect cropRaw, boolean valid) {
        ViewGeom g = viewGeom;
        if (g == null) return;
        float[] crop = FrameGeometry.rawRectToUpright(cropRaw.left, cropRaw.top, cropRaw.right, cropRaw.bottom,
                rawW, rawH, rotation);
        if (crop[2] - crop[0] < 1 || crop[3] - crop[1] < 1) {
            crop = new float[]{0, 0, FrameGeometry.uprightWidth(rawW, rawH, rotation),
                    FrameGeometry.uprightHeight(rawW, rawH, rotation)};
        }
        final boolean mirror = true; // front camera: PreviewView shows a mirrored image

        Rect bb = face.getBoundingBox();
        PointF p1 = toView(bb.left, bb.top, crop, g, mirror);
        PointF p2 = toView(bb.right, bb.bottom, crop, g, mirror);
        RectF faceRect = new RectF(Math.min(p1.x, p2.x), Math.min(p1.y, p2.y),
                Math.max(p1.x, p2.x), Math.max(p1.y, p2.y));

        List<PointF> rc = new ArrayList<>(), lc = new ArrayList<>();
        for (int k = 0; k < rCont.length / 2; k++) rc.add(toView(rCont[2 * k], rCont[2 * k + 1], crop, g, mirror));
        for (int k = 0; k < lCont.length / 2; k++) lc.add(toView(lCont[2 * k], lCont[2 * k + 1], crop, g, mirror));

        List<PointF> pupils = new ArrayList<>();
        if (rightRes.valid) pupils.add(toView(rightRes.cxUp, rightRes.cyUp, crop, g, mirror));
        if (leftRes.valid) pupils.add(toView(leftRes.cxUp, leftRes.cyUp, crop, g, mirror));

        float irisR = 0.2f * 0.5f * (rightRes.eyeWidth + leftRes.eyeWidth)
                * Math.max(g.pw / (crop[2] - crop[0]), g.ph / (crop[3] - crop[1]));

        List<List<PointF>> contours = new ArrayList<>();
        contours.add(rc);
        contours.add(lc);
        postOverlay(new FaceOverlayView.OverlayData(faceRect, contours, pupils, irisR, valid));
    }

    private static PointF toView(float u, float v, float[] crop, ViewGeom g, boolean mirror) {
        float[] o = new float[2];
        FrameGeometry.uprightToView(u, v, crop, g.pw, g.ph, mirror, o);
        return new PointF(o[0] + g.offX, o[1] + g.offY);
    }

    private void postOverlay(FaceOverlayView.OverlayData d) {
        if (overlayView == null) return;
        overlayView.post(() -> overlayView.setData(d));
    }

    // ---------------------------------------------------------------------------------------
    // Demo mode (no camera): synthetic fixations/saccades in image-normalised coordinates
    // ---------------------------------------------------------------------------------------

    private void demoTick() {
        long now = SystemClock.elapsedRealtime();
        if (now >= demoNextSaccade) {
            demoTx = 0.1f + 0.8f * rnd.nextFloat();
            demoTy = 0.1f + 0.8f * rnd.nextFloat();
            demoNextSaccade = now + 150 + rnd.nextInt(450);
        }
        demoX += 0.6f * (demoTx - demoX) + (float) rnd.nextGaussian() * 0.005f;
        demoY += 0.6f * (demoTy - demoY) + (float) rnd.nextGaussian() * 0.005f;
        synchronized (winLock) {
            if (inWindow && now >= winStart && now <= winEnd) {
                winSamples.add(new GazeSample(now, demoX, demoY, true));
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------

    private static PointF pos(List<FaceMeshPoint> pts, int idx) {
        PointF3D p = pts.get(idx).getPosition();
        return new PointF(p.getX(), p.getY());
    }

    private static float[] contour(List<FaceMeshPoint> pts, int[] idx) {
        float[] out = new float[idx.length * 2];
        for (int k = 0; k < idx.length; k++) {
            PointF3D p = pts.get(idx[k]).getPosition();
            out[2 * k] = p.getX();
            out[2 * k + 1] = p.getY();
        }
        return out;
    }
}
