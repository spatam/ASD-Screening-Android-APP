package it.unict.dmi.asdscreening;

/**
 * Pure-Java helpers to move between the RAW camera buffer coordinate system and the
 * UPRIGHT coordinate system in which ML Kit returns landmarks when the InputImage is built
 * with {@code InputImage.fromMediaImage(image, rotationDegrees)}.
 *
 * rotationDegrees = clockwise rotation that makes the raw buffer upright
 * (value of ImageProxy.getImageInfo().getRotationDegrees()).
 *
 * Continuous coordinates are used: a raw buffer of W x H pixels spans [0,W] x [0,H].
 */
public final class FrameGeometry {

    private FrameGeometry() {}

    /** Width of the upright frame. */
    public static int uprightWidth(int rawW, int rawH, int rotation) {
        return (rotation == 90 || rotation == 270) ? rawH : rawW;
    }

    /** Height of the upright frame. */
    public static int uprightHeight(int rawW, int rawH, int rotation) {
        return (rotation == 90 || rotation == 270) ? rawW : rawH;
    }

    /** Raw buffer point -> upright point. out = {u, v}. */
    public static void rawToUpright(float x, float y, int rawW, int rawH, int rotation, float[] out) {
        switch (rotation) {
            case 90:  out[0] = rawH - y; out[1] = x;        break;
            case 180: out[0] = rawW - x; out[1] = rawH - y; break;
            case 270: out[0] = y;        out[1] = rawW - x; break;
            default:  out[0] = x;        out[1] = y;        break;
        }
    }

    /** Upright point -> raw buffer point. out = {x, y}. */
    public static void uprightToRaw(float u, float v, int rawW, int rawH, int rotation, float[] out) {
        switch (rotation) {
            case 90:  out[0] = v;        out[1] = rawH - u; break;
            case 180: out[0] = rawW - u; out[1] = rawH - v; break;
            case 270: out[0] = rawW - v; out[1] = u;        break;
            default:  out[0] = u;        out[1] = v;        break;
        }
    }

    /**
     * Raw rectangle (left, top, right, bottom) -> upright rectangle, returned as
     * {left, top, right, bottom}.
     */
    public static float[] rawRectToUpright(float l, float t, float r, float b,
                                           int rawW, int rawH, int rotation) {
        float[] p1 = new float[2];
        float[] p2 = new float[2];
        rawToUpright(l, t, rawW, rawH, rotation, p1);
        rawToUpright(r, b, rawW, rawH, rotation, p2);
        return new float[]{
                Math.min(p1[0], p2[0]), Math.min(p1[1], p2[1]),
                Math.max(p1[0], p2[0]), Math.max(p1[1], p2[1])
        };
    }

    /**
     * Maps an upright-frame point into the coordinate system of a PreviewView of size
     * viewW x viewH, given the (upright) crop rect of the analysis frame that corresponds
     * to what the PreviewView shows.
     *
     * When the Preview and ImageAnalysis use cases are bound in the same UseCaseGroup with
     * the PreviewView's ViewPort, the crop rect has exactly the view's aspect ratio and
     * the mapping is a pure scale. Otherwise this reproduces PreviewView FILL_CENTER.
     *
     * @param crop   upright crop rect {left, top, right, bottom}
     * @param mirror true for the front camera (PreviewView mirrors it horizontally)
     * @param out    {xView, yView}
     */
    public static void uprightToView(float u, float v, float[] crop, int viewW, int viewH,
                                     boolean mirror, float[] out) {
        float cw = crop[2] - crop[0];
        float ch = crop[3] - crop[1];
        float scale = Math.max(viewW / cw, viewH / ch);
        float cx = (crop[0] + crop[2]) * 0.5f;
        float cy = (crop[1] + crop[3]) * 0.5f;
        float xv = (u - cx) * scale + viewW * 0.5f;
        float yv = (v - cy) * scale + viewH * 0.5f;
        if (mirror) xv = viewW - xv;
        out[0] = xv;
        out[1] = yv;
    }
}
