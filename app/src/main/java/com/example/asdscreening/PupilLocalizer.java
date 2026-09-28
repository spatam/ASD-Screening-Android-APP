package com.example.asdscreening;

/**
 * Locates the iris/pupil centre inside one eye using the luminance (Y) plane of the camera
 * frame and the eyelid contour returned by ML Kit Face Mesh.
 *
 * Why this is needed: ML Kit Face Mesh (468 points) has NO iris landmarks. The eye-contour
 * landmarks follow the eyelids and eye corners, i.e. the head, not the eyeball rotation.
 * Their centroid is therefore almost constant while the subject explores an image, which is
 * why the previous implementation produced the same fixations - and the same p(ASD) - for
 * every subject.
 *
 * Method (deterministic, ~0.2 ms per eye):
 *  1. Build an eye-aligned patch (PW x PH) from the raw Y plane with bilinear sampling.
 *     The x axis goes from the image-left eye corner to the image-right one, so the patch is
 *     invariant to head roll and to the camera sensor rotation.
 *  2. Rasterise the (slightly shrunk) eyelid polygon as a mask: lashes, lid margins and
 *     corner shadows are excluded.
 *  3. Coarse search: masked box-mean via integral images, the darkest iris-sized window wins.
 *  4. Refinement: darkness-weighted centroid inside an iris-radius disc (sub-pixel).
 *
 * Output is expressed in the eye frame, normalised by the eye width, so it is independent of
 * the subject's distance from the phone: (ex, ey) = iris offset from the eye-corner midpoint.
 *
 * Not thread-safe (internal buffers are reused): use one instance per eye per thread.
 */
public final class PupilLocalizer {

    public static final class Result {
        public boolean valid;
        /** Iris centre in UPRIGHT frame coordinates (same space as ML Kit landmarks). */
        public float cxUp, cyUp;
        /** Iris offset in the eye frame, normalised by eye width. */
        public float ex, ey;
        /** Eyelid opening / eye width (blink when small). */
        public float openness;
        /** Median(eye) - darkest window mean, in grey levels. */
        public float contrast;
        /** Eye width in upright pixels. */
        public float eyeWidth;
    }

    // Patch geometry (in units of eye width w)
    private static final int PW = 64;
    private static final int PH = 32;
    private static final float PATCH_W_REL = 1.10f;                 // patch spans 1.1 w
    private static final float IRIS_R_REL = 0.20f;                  // iris radius ~ 0.2 w
    private static final float MASK_SHRINK_X = 0.92f;
    private static final float MASK_SHRINK_Y = 0.80f;

    // Validity thresholds
    public static float MIN_EYE_WIDTH_PX = 10f;
    public static float BLINK_OPENNESS = 0.13f;
    public static float MIN_CONTRAST = 6f;
    private static final int MIN_MASK_PIXELS = 40;

    private final float[] patch = new float[PW * PH];
    private final boolean[] mask = new boolean[PW * PH];
    private final double[] intI = new double[(PW + 1) * (PH + 1)];
    private final int[] intM = new int[(PW + 1) * (PH + 1)];
    private final float[] tmp2 = new float[2];
    private final int[] hist = new int[256];

    /**
     * @param luma        Y plane bytes (as copied from ImageProxy plane 0)
     * @param rowStride   plane row stride
     * @param pixelStride plane pixel stride
     * @param rawW        raw buffer width
     * @param rawH        raw buffer height
     * @param rotation    rotation degrees (0/90/180/270)
     * @param ax,ay       image-left eye corner (upright coords)
     * @param bx,by       image-right eye corner (upright coords)
     * @param contour     eyelid contour, interleaved x,y in upright coords (closed polygon)
     */
    public Result locate(byte[] luma, int rowStride, int pixelStride, int rawW, int rawH,
                         int rotation, float ax, float ay, float bx, float by,
                         float[] contour, Result out) {
        if (out == null) out = new Result();
        out.valid = false;

        float dx = bx - ax, dy = by - ay;
        float w = (float) Math.sqrt(dx * dx + dy * dy);
        out.eyeWidth = w;
        if (w < MIN_EYE_WIDTH_PX) return out;

        float uxX = dx / w, uxY = dy / w;       // eye x axis
        float uyX = -uxY, uyY = uxX;            // eye y axis (points "down" in the image)
        float ox = (ax + bx) * 0.5f, oy = (ay + by) * 0.5f;
        float s = PATCH_W_REL * w / PW;         // upright pixels per patch pixel

        // --- contour in eye frame (normalised by w), openness, shrunk polygon in patch px
        int nc = contour.length / 2;
        float[] poly = new float[nc * 2];
        float minEy = Float.MAX_VALUE, maxEy = -Float.MAX_VALUE;
        for (int k = 0; k < nc; k++) {
            float px = contour[2 * k] - ox, py = contour[2 * k + 1] - oy;
            float ex = (px * uxX + py * uxY);
            float ey = (px * uyX + py * uyY);
            minEy = Math.min(minEy, ey);
            maxEy = Math.max(maxEy, ey);
            poly[2 * k] = ex * MASK_SHRINK_X / s + PW * 0.5f;
            poly[2 * k + 1] = ey * MASK_SHRINK_Y / s + PH * 0.5f;
        }
        out.openness = (maxEy - minEy) / w;
        if (out.openness < BLINK_OPENNESS) return out;

        // --- sample patch + mask
        int maskCount = 0;
        java.util.Arrays.fill(hist, 0);
        for (int j = 0; j < PH; j++) {
            float ey = (j + 0.5f - PH * 0.5f) * s;
            for (int i = 0; i < PW; i++) {
                float ex = (i + 0.5f - PW * 0.5f) * s;
                float u = ox + ex * uxX + ey * uyX;
                float v = oy + ex * uxY + ey * uyY;
                FrameGeometry.uprightToRaw(u, v, rawW, rawH, rotation, tmp2);
                float val = sampleBilinear(luma, rowStride, pixelStride, rawW, rawH, tmp2[0], tmp2[1]);
                int idx = j * PW + i;
                patch[idx] = val;
                boolean in = pointInPolygon(i + 0.5f, j + 0.5f, poly);
                mask[idx] = in;
                if (in) {
                    maskCount++;
                    hist[Math.max(0, Math.min(255, (int) val))]++;
                }
            }
        }
        if (maskCount < MIN_MASK_PIXELS) return out;
        float median = histPercentile(hist, maskCount, 0.5f);

        // --- integral images of masked intensity and mask
        int stride = PW + 1;
        for (int j = 0; j <= PH; j++) { intI[j * stride] = 0; intM[j * stride] = 0; }
        for (int i = 0; i <= PW; i++) { intI[i] = 0; intM[i] = 0; }
        for (int j = 1; j <= PH; j++) {
            double rowI = 0; int rowM = 0;
            for (int i = 1; i <= PW; i++) {
                int idx = (j - 1) * PW + (i - 1);
                if (mask[idx]) { rowI += patch[idx]; rowM++; }
                intI[j * stride + i] = intI[(j - 1) * stride + i] + rowI;
                intM[j * stride + i] = intM[(j - 1) * stride + i] + rowM;
            }
        }

        // --- coarse search: darkest masked box of iris size
        float rPatch = IRIS_R_REL * w / s;
        int h = Math.max(2, Math.round(0.65f * rPatch));
        int minCover = (int) (0.30f * (2 * h + 1) * (2 * h + 1));
        float best = Float.MAX_VALUE;
        int bestI = -1, bestJ = -1;
        for (int j = 0; j < PH; j++) {
            int y0 = Math.max(0, j - h), y1 = Math.min(PH, j + h + 1);
            for (int i = 0; i < PW; i++) {
                if (!mask[j * PW + i]) continue;
                int x0 = Math.max(0, i - h), x1 = Math.min(PW, i + h + 1);
                int m = intM[y1 * stride + x1] - intM[y0 * stride + x1]
                        - intM[y1 * stride + x0] + intM[y0 * stride + x0];
                if (m < minCover) continue;
                double si = intI[y1 * stride + x1] - intI[y0 * stride + x1]
                        - intI[y1 * stride + x0] + intI[y0 * stride + x0];
                float mean = (float) (si / m);
                if (mean < best) { best = mean; bestI = i; bestJ = j; }
            }
        }
        if (bestI < 0) return out;
        out.contrast = median - best;
        if (out.contrast < MIN_CONTRAST) return out;

        // --- refinement: darkness-weighted centroid in iris disc
        double sw = 0, sx = 0, sy = 0;
        float r2 = rPatch * rPatch;
        int jr0 = Math.max(0, (int) Math.floor(bestJ - rPatch)), jr1 = Math.min(PH - 1, (int) Math.ceil(bestJ + rPatch));
        int ir0 = Math.max(0, (int) Math.floor(bestI - rPatch)), ir1 = Math.min(PW - 1, (int) Math.ceil(bestI + rPatch));
        for (int j = jr0; j <= jr1; j++) {
            for (int i = ir0; i <= ir1; i++) {
                int idx = j * PW + i;
                if (!mask[idx]) continue;
                float ddx = i - bestI, ddy = j - bestJ;
                if (ddx * ddx + ddy * ddy > r2) continue;
                float wgt = median - patch[idx];
                if (wgt <= 0) continue;
                sw += wgt; sx += wgt * (i + 0.5); sy += wgt * (j + 0.5);
            }
        }
        float cpx, cpy;
        if (sw > 0) { cpx = (float) (sx / sw); cpy = (float) (sy / sw); }
        else { cpx = bestI + 0.5f; cpy = bestJ + 0.5f; }

        float exPix = (cpx - PW * 0.5f) * s;     // upright px along eye x axis
        float eyPix = (cpy - PH * 0.5f) * s;
        out.ex = exPix / w;
        out.ey = eyPix / w;
        out.cxUp = ox + exPix * uxX + eyPix * uyX;
        out.cyUp = oy + exPix * uxY + eyPix * uyY;
        out.valid = true;
        return out;
    }

    // ------------------------------------------------------------------------------------

    static float sampleBilinear(byte[] y, int rs, int ps, int w, int h, float x, float yy) {
        float fx = x - 0.5f, fy = yy - 0.5f;      // pixel centres at +0.5
        if (fx < 0) fx = 0; if (fy < 0) fy = 0;
        if (fx > w - 1.001f) fx = w - 1.001f;
        if (fy > h - 1.001f) fy = h - 1.001f;
        int x0 = (int) fx, y0 = (int) fy;
        float ax = fx - x0, ay = fy - y0;
        int i00 = y0 * rs + x0 * ps;
        int i10 = i00 + ps;
        int i01 = i00 + rs;
        int i11 = i01 + ps;
        int n = y.length;
        float v00 = (i00 < n) ? (y[i00] & 0xFF) : 0;
        float v10 = (i10 < n) ? (y[i10] & 0xFF) : v00;
        float v01 = (i01 < n) ? (y[i01] & 0xFF) : v00;
        float v11 = (i11 < n) ? (y[i11] & 0xFF) : v01;
        return (v00 * (1 - ax) + v10 * ax) * (1 - ay) + (v01 * (1 - ax) + v11 * ax) * ay;
    }

    static boolean pointInPolygon(float x, float y, float[] poly) {
        boolean inside = false;
        int n = poly.length / 2;
        for (int a = 0, b = n - 1; a < n; b = a++) {
            float xa = poly[2 * a], ya = poly[2 * a + 1];
            float xb = poly[2 * b], yb = poly[2 * b + 1];
            if ((ya > y) != (yb > y)) {
                float xCross = (xb - xa) * (y - ya) / (yb - ya) + xa;
                if (x < xCross) inside = !inside;
            }
        }
        return inside;
    }

    private static float histPercentile(int[] hist, int total, float q) {
        int target = (int) (q * total);
        int acc = 0;
        for (int i = 0; i < 256; i++) {
            acc += hist[i];
            if (acc > target) return i;
        }
        return 255;
    }
}
