package com.example.asdscreening;

import android.graphics.Bitmap;
import android.graphics.Color;

import java.util.ArrayList;
import java.util.List;

public class Preprocessing {

    private static final int H = 224;
    private static final int W = 224;
    private static final int T = 25;

    private static final float[] IMAGENET_MEAN = new float[]{0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD  = new float[]{0.229f, 0.224f, 0.225f};

    /**
     * gaze_seq: (1,25,3) float32 flattened.
     * Steps:
     *  - x,y local min-max -> [0,1] (if max==min -> 0.5)
     *  - duration log1p / 10
     *  - pad/truncate to 25 with zeros
     */
    public static float[] buildGazeSeq(List<Fixation> fixations) {
        List<Fixation> fx = (fixations == null) ? new ArrayList<>() : fixations;
        int n = Math.min(fx.size(), T);

        float[] x = new float[n];
        float[] y = new float[n];
        float[] d = new float[n];

        for (int i = 0; i < n; i++) {
            x[i] = fx.get(i).xScreen;
            y[i] = fx.get(i).yScreen;
            d[i] = fx.get(i).durationMs;
        }

        float xMin = min(x), xMax = max(x);
        float yMin = min(y), yMax = max(y);

        float[] out = new float[T * 3]; // row-major (t,3)
        for (int i = 0; i < T; i++) {
            float xn = 0f, yn = 0f, dn = 0f;
            if (i < n) {
                xn = normMinMax(x[i], xMin, xMax);
                yn = normMinMax(y[i], yMin, yMax);
                dn = (float) (Math.log1p(Math.max(0.0, d[i])) / 10.0);
            }
            out[i * 3] = xn;
            out[i * 3 + 1] = yn;
            out[i * 3 + 2] = dn;
        }
        return out;
    }

    /**
     * stimulus_image: (1,3,224,224) float32 flattened CHW.
     * Uses fixations mapped to nominal 1280x1024 coordinates as provided.
     */
    public static float[] buildStimulusImageTensor(Bitmap stimulusBmp, List<Fixation> fixations) {
        // Step A: resize, to RGB float [0,1] CHW
        Bitmap resized = Bitmap.createScaledBitmap(stimulusBmp, W, H, true);
        float[] img = bitmapToCHW01(resized); // 3*H*W

        // Step B: heatmap 224x224
        float[] hmap = new float[H * W];
        if (fixations != null) {
            for (Fixation f : fixations) {
                int cx = (int) clamp((f.xScreen / 1280.0f) * 223.0f, 0f, 223f);
                int cy = (int) clamp((f.yScreen / 1024.0f) * 223.0f, 0f, 223f);
                float w = (float) Math.log1p(Math.max(0.0, f.durationMs));
                hmap[cy * W + cx] += w;
            }
        }
        // gaussian filter sigma=10
        hmap = Gaussian.gaussianBlur(hmap, H, W, 10.0f);

        // normalize to [0,1]
        float m = max(hmap);
        if (m > 1e-12f) {
            for (int i = 0; i < hmap.length; i++) hmap[i] /= m;
        }

        // Step C: Jet colormap -> heatmap_rgb CHW [0,1]
        float[] heat = JetColorMap.jetToCHW(hmap, H, W);

        // Step D: Blend
        float[] blended = new float[3 * H * W];
        int plane = H * W;
        for (int i = 0; i < 3 * plane; i++) {
            blended[i] = 0.55f * img[i] + 0.45f * heat[i];
        }

        // Step E: ImageNet normalization per channel
        for (int c = 0; c < 3; c++) {
            float mean = IMAGENET_MEAN[c];
            float std = IMAGENET_STD[c];
            int off = c * plane;
            for (int i = 0; i < plane; i++) {
                blended[off + i] = (blended[off + i] - mean) / std;
            }
        }

        return blended;
    }

    private static float[] bitmapToCHW01(Bitmap bmp) {
        int w = bmp.getWidth();
        int h = bmp.getHeight();
        int[] pixels = new int[w * h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);

        float[] out = new float[3 * w * h];
        int plane = w * h;
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            float r = ((p >> 16) & 0xFF) / 255.0f;
            float g = ((p >> 8) & 0xFF) / 255.0f;
            float b = (p & 0xFF) / 255.0f;
            out[i] = r;
            out[plane + i] = g;
            out[2 * plane + i] = b;
        }
        return out;
    }

    private static float normMinMax(float v, float vmin, float vmax) {
        if (Math.abs(vmax - vmin) < 1e-9f) return 0.5f;
        float x = (v - vmin) / (vmax - vmin);
        if (Float.isNaN(x) || Float.isInfinite(x)) return 0.5f;
        return clamp(x, 0f, 1f);
    }

    private static float min(float[] a) {
        if (a == null || a.length == 0) return 0f;
        float m = a[0];
        for (float v : a) m = Math.min(m, v);
        return m;
    }

    private static float max(float[] a) {
        if (a == null || a.length == 0) return 0f;
        float m = a[0];
        for (float v : a) m = Math.max(m, v);
        return m;
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
