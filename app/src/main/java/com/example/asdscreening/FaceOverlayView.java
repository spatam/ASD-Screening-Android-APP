package com.example.asdscreening;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Overlay drawn on top of the camera preview:
 *  - face bounding box (coloured by running p(ASD))
 *  - eyelid contours (blue polylines, from Face Mesh)
 *  - detected iris/pupil (green ring + centre dot) - the REAL pupil position used for gaze,
 *    not the eye-socket centroid.
 * All coordinates are in this view's coordinate system (computed by CameraGazeTracker).
 */
public class FaceOverlayView extends View {

    /** Immutable payload passed from the camera thread to the UI. */
    public static class OverlayData {
        public final RectF faceRect;
        public final List<List<PointF>> eyeContours;
        public final List<PointF> pupils;
        public final float irisRadiusPx;
        public final boolean trackingValid;

        public OverlayData(RectF faceRect, List<List<PointF>> eyeContours, List<PointF> pupils,
                           float irisRadiusPx, boolean trackingValid) {
            this.faceRect = faceRect;
            List<List<PointF>> c = new ArrayList<>();
            if (eyeContours != null) for (List<PointF> l : eyeContours) c.add(Collections.unmodifiableList(new ArrayList<>(l)));
            this.eyeContours = Collections.unmodifiableList(c);
            this.pupils = pupils != null ? Collections.unmodifiableList(new ArrayList<>(pupils)) : Collections.emptyList();
            this.irisRadiusPx = irisRadiusPx;
            this.trackingValid = trackingValid;
        }
    }

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint contourPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pupilPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint irisPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();

    private volatile @Nullable OverlayData data;
    private volatile float asdProbability = Float.NaN;
    private volatile @Nullable String statusLabel;

    public FaceOverlayView(Context context) { super(context); init(); }
    public FaceOverlayView(Context context, @Nullable AttributeSet attrs) { super(context, attrs); init(); }
    public FaceOverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) { super(context, attrs, defStyleAttr); init(); }

    private void init() {
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(dp(2));

        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(dp(12));
        textPaint.setShadowLayer(dp(2), 0, 0, Color.BLACK);

        contourPaint.setStyle(Paint.Style.STROKE);
        contourPaint.setStrokeWidth(dp(1.2f));
        contourPaint.setColor(Color.rgb(40, 120, 255));

        pupilPaint.setStyle(Paint.Style.FILL);
        pupilPaint.setColor(Color.GREEN);

        irisPaint.setStyle(Paint.Style.STROKE);
        irisPaint.setStrokeWidth(dp(1.2f));
        irisPaint.setColor(Color.GREEN);
    }

    private float dp(float v) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    public void setData(@Nullable OverlayData d) { data = d; postInvalidateOnAnimation(); }

    public void setAsdProbability(float p) { asdProbability = p; postInvalidateOnAnimation(); }

    /** Optional short label (e.g. "CALIBRATING") drawn at the top of the overlay. */
    public void setStatusLabel(@Nullable String s) { statusLabel = s; postInvalidateOnAnimation(); }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        String label = statusLabel;
        if (label != null) canvas.drawText(label, dp(4), dp(14), textPaint);

        OverlayData d = data;
        if (d == null) return;

        int color = Color.GRAY;
        float p = asdProbability;
        if (!Float.isNaN(p)) {
            if (p >= 0.66f) color = Color.RED;
            else if (p < 0.33f) color = Color.GREEN;
        }
        if (!d.trackingValid) color = Color.YELLOW;   // face found but eyes not trackable
        boxPaint.setColor(color);

        if (d.faceRect != null) {
            canvas.drawRect(d.faceRect, boxPaint);
            if (!Float.isNaN(p)) {
                String s = String.format(Locale.US, "ASD: %.1f%%", 100f * p);
                canvas.drawText(s, d.faceRect.left, Math.max(dp(28), d.faceRect.top - dp(4)), textPaint);
            }
        }

        for (List<PointF> c : d.eyeContours) {
            if (c.size() < 3) continue;
            path.reset();
            path.moveTo(c.get(0).x, c.get(0).y);
            for (int i = 1; i < c.size(); i++) path.lineTo(c.get(i).x, c.get(i).y);
            path.close();
            canvas.drawPath(path, contourPaint);
        }

        float rDot = Math.max(dp(1.5f), d.irisRadiusPx * 0.25f);
        for (PointF q : d.pupils) {
            if (d.irisRadiusPx > dp(2)) canvas.drawCircle(q.x, q.y, d.irisRadiusPx, irisPaint);
            canvas.drawCircle(q.x, q.y, rDot, pupilPaint);
        }
    }
}
