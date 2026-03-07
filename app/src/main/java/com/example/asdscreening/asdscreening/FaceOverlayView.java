package com.example.asdscreening;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Overlay view:
 *  - face bounding box (colored by ASD probability)
 *  - eye mesh points (blue)
 *  - eye centers (green)
 */
public class FaceOverlayView extends View {

    /** Immutable payload passed from camera thread to UI. */
    public static class OverlayData {
        public final RectF faceRect;            // in view coordinates
        public final List<PointF> meshPoints;   // in view coordinates
        public final List<PointF> gazePoints;   // in view coordinates

        /** Preferred ctor: (RectF, mesh, gaze). */
        public OverlayData(RectF faceRect, List<PointF> meshPoints, List<PointF> gazePoints) {
            this.faceRect = faceRect;
            this.meshPoints = meshPoints != null
                    ? Collections.unmodifiableList(new ArrayList<>(meshPoints))
                    : Collections.emptyList();
            this.gazePoints = gazePoints != null
                    ? Collections.unmodifiableList(new ArrayList<>(gazePoints))
                    : Collections.emptyList();
        }

        /** Backward-compatible overload: no gaze points. */
        public OverlayData(RectF faceRect, List<PointF> meshPoints) {
            this(faceRect, meshPoints, null);
        }
    }

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint meshPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gazePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private volatile @Nullable OverlayData data;
    private volatile float asdProbability = Float.NaN; // 0..1

    public FaceOverlayView(Context context) {
        super(context);
        init();
    }

    public FaceOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public FaceOverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        float stroke = dp(3);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(stroke);

        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(dp(14));
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setShadowLayer(dp(2), 0, 0, Color.BLACK);

        // Blue mesh points (eye mesh)
        meshPaint.setStyle(Paint.Style.FILL);
        meshPaint.setColor(Color.BLUE);

        // Green gaze points (eye centers)
        gazePaint.setStyle(Paint.Style.FILL);
        gazePaint.setColor(Color.GREEN);
    }

    private float dp(float value) {
        return TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP,
                value,
                getResources().getDisplayMetrics()
        );
    }

    /** Called from UI thread. */
    public void setData(@Nullable OverlayData d) {
        this.data = d;
        postInvalidateOnAnimation();
    }

    /** Called from UI thread. */
    public void setAsdProbability(float p) {
        this.asdProbability = p;
        postInvalidateOnAnimation();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        OverlayData d = data;
        if (d == null) return;

        // Pick box color based on p(ASD)
        int color = Color.GRAY;
        if (!Float.isNaN(asdProbability)) {
            if (asdProbability >= 0.66f) {
                color = Color.RED; // ASD likely
            } else if (asdProbability >= 0.33f) {
                color = Color.GRAY; // uncertain
            } else {
                color = Color.GREEN; // ASD unlikely
            }
        }
        boxPaint.setColor(color);

        // Face box
        if (d.faceRect != null) {
            canvas.drawRect(d.faceRect, boxPaint);

            if (!Float.isNaN(asdProbability)) {
                String s = String.format("ASD: %.1f%%", 100f * asdProbability);
                float x = d.faceRect.left;
                float y = Math.max(dp(16), d.faceRect.top - dp(6));
                canvas.drawText(s, x, y, textPaint);
            }
        }

        // Mesh points (blue)
        float rMesh = dp(1.6f);
        for (PointF p : d.meshPoints) {
            canvas.drawCircle(p.x, p.y, rMesh, meshPaint);
        }

        // Gaze points (green, larger)
        float rGaze = dp(3.2f);
        for (PointF p : d.gazePoints) {
            canvas.drawCircle(p.x, p.y, rGaze, gazePaint);
        }
    }
}
