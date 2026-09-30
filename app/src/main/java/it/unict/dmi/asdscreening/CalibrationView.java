package it.unict.dmi.asdscreening;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.TypedValue;
import android.view.View;
import android.view.animation.DecelerateInterpolator;

/**
 * Full-screen transparent view that draws the calibration target.
 * Targets are given in SCREEN coordinates (same space as the gaze predictions and the
 * stimulus rectangle), and converted to local coordinates here.
 *
 * The target shrinks from a large ring to a small dot: this attracts the gaze and makes the
 * subject fixate its centre, which is when samples are collected.
 */
public class CalibrationView extends View {

    private final Paint ring = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dot = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);

    private boolean visible = false;
    private float sx, sy;
    private float anim = 1f; // 1 = large ring, 0 = small
    private ValueAnimator animator;
    private String message;

    public CalibrationView(Context ctx) {
        super(ctx);
        ring.setStyle(Paint.Style.STROKE);
        ring.setStrokeWidth(dp(3));
        ring.setColor(Color.WHITE);
        dot.setStyle(Paint.Style.FILL);
        dot.setColor(Color.RED);
        text.setColor(Color.WHITE);
        text.setTextSize(dp(18));
        text.setTextAlign(Paint.Align.CENTER);
        text.setShadowLayer(dp(2), 0, 0, Color.BLACK);
        setClickable(false);
        setFocusable(false);
    }

    private float dp(float v) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    public void showTarget(float screenX, float screenY, long shrinkMs) {
        sx = screenX; sy = screenY; visible = true;
        if (animator != null) animator.cancel();
        animator = ValueAnimator.ofFloat(1f, 0f);
        animator.setDuration(shrinkMs);
        animator.setInterpolator(new DecelerateInterpolator());
        animator.addUpdateListener(a -> { anim = (float) a.getAnimatedValue(); invalidate(); });
        animator.start();
    }

    public void setMessage(String m) { message = m; invalidate(); }

    public void hideTarget() {
        visible = false;
        if (animator != null) animator.cancel();
        invalidate();
    }

    @Override
    protected void onDraw(Canvas c) {
        super.onDraw(c);
        int[] loc = new int[2];
        getLocationOnScreen(loc);
        if (message != null) c.drawText(message, getWidth() / 2f, dp(80), text);
        if (!visible) return;
        float x = sx - loc[0], y = sy - loc[1];
        float r = dp(6) + anim * dp(30);
        c.drawCircle(x, y, r, ring);
        c.drawCircle(x, y, dp(4), dot);
    }
}
