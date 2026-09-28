package com.example.asdscreening;

/**
 * Jet colormap implementation mapping v in [0,1] to RGB in [0,1].
 * Output in CHW order (3, H, W).
 */
public class JetColorMap {

    public static float[] jetToCHW(float[] hmap, int H, int W) {
        int plane = H * W;
        float[] out = new float[3 * plane];

        for (int i = 0; i < plane; i++) {
            float v = clamp(hmap[i], 0f, 1f);
            float[] rgb = jet(v);
            out[i] = rgb[0];
            out[plane + i] = rgb[1];
            out[2 * plane + i] = rgb[2];
        }
        return out;
    }

    // Classic "jet" style ramps
    private static float[] jet(float v) {
        // piecewise linear approximation
        float r = clamp(1.5f - Math.abs(4f * v - 3f), 0f, 1f);
        float g = clamp(1.5f - Math.abs(4f * v - 2f), 0f, 1f);
        float b = clamp(1.5f - Math.abs(4f * v - 1f), 0f, 1f);
        return new float[]{r, g, b};
    }

    private static float clamp(float x, float lo, float hi) {
        return Math.max(lo, Math.min(hi, x));
    }
}
