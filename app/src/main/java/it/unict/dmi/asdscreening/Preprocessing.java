package it.unict.dmi.asdscreening;

import android.graphics.Bitmap;

import java.util.Collections;
import java.util.List;

/**
 * Builds the two model inputs exactly as the Python training code does
 * ({@code asd_gaze/preprocessing.py}). {@code PreprocessingParityTest} checks both functions
 * against values produced by that code.
 */
public final class Preprocessing {

    public static final int SIZE = 224;
    public static final int SEQ_LEN = 25;

    /**
     * Screen of the Tobii T120 used to record Saliency4ASD. The training code divides stimulus
     * pixel coordinates by these values when it places fixations on the 224x224 heatmap, so a
     * 1024x768 picture only covers part of the grid. The models learned that mapping; keep it.
     */
    public static final float SCREEN_WIDTH = 1280f;
    public static final float SCREEN_HEIGHT = 1024f;

    public static final float BLEND_ALPHA = 0.55f;
    public static final float HEATMAP_SIGMA = 10f;

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private Preprocessing() {}

    /**
     * Gaze sequence, shape (25, 3) flattened row-major: x and y min-max scaled over all
     * fixations, log1p(duration ms) / 10, then truncated or zero-padded to 25 rows.
     */
    public static float[] buildGazeSeq(List<Fixation> fixations) {
        List<Fixation> fx = fixations == null ? Collections.emptyList() : fixations;
        float xMin = Float.MAX_VALUE, xMax = -Float.MAX_VALUE;
        float yMin = Float.MAX_VALUE, yMax = -Float.MAX_VALUE;
        // The scaling range covers every fixation, including those past row 25.
        for (Fixation f : fx) {
            xMin = Math.min(xMin, f.x);
            xMax = Math.max(xMax, f.x);
            yMin = Math.min(yMin, f.y);
            yMax = Math.max(yMax, f.y);
        }

        float[] out = new float[SEQ_LEN * 3];
        int n = Math.min(fx.size(), SEQ_LEN);
        for (int i = 0; i < n; i++) {
            Fixation f = fx.get(i);
            out[i * 3] = minMax(f.x, xMin, xMax);
            out[i * 3 + 1] = minMax(f.y, yMin, yMax);
            out[i * 3 + 2] = (float) (Math.log1p(Math.max(0.0, f.durationMs)) / 10.0);
        }
        return out;
    }

    /** Stimulus tensor, shape (3, 224, 224) flattened CHW. */
    public static float[] buildStimulusImageTensor(Bitmap stimulus, List<Fixation> fixations) {
        int w = stimulus.getWidth();
        int h = stimulus.getHeight();
        int[] argb = new int[w * h];
        stimulus.getPixels(argb, 0, w, 0, 0, w, h);
        return buildStimulusImageTensor(argb, w, h, fixations);
    }

    /**
     * Pillow-compatible bilinear resize to 224x224, duration-weighted Gaussian heatmap of the
     * fixations (sigma 10, peak-normalised) coloured with a jet map, blend
     * 0.55 * image + 0.45 * heatmap, then ImageNet normalisation.
     */
    public static float[] buildStimulusImageTensor(int[] argb, int width, int height, List<Fixation> fixations) {
        int plane = SIZE * SIZE;
        int[] resized = PilResize.bilinear(argb, width, height, SIZE, SIZE);

        float[] heatmap = new float[plane];
        if (fixations != null) {
            for (Fixation f : fixations) {
                int cx = (int) clamp(f.x / SCREEN_WIDTH * (SIZE - 1), 0f, SIZE - 1);
                int cy = (int) clamp(f.y / SCREEN_HEIGHT * (SIZE - 1), 0f, SIZE - 1);
                heatmap[cy * SIZE + cx] += (float) Math.log1p(Math.max(0.0, f.durationMs));
            }
        }
        heatmap = Gaussian.gaussianBlur(heatmap, SIZE, SIZE, HEATMAP_SIGMA);
        float peak = 0f;
        for (float v : heatmap) peak = Math.max(peak, v);
        if (peak > 0f) {
            for (int i = 0; i < plane; i++) heatmap[i] /= peak;
        }
        float[] colour = JetColorMap.jetToCHW(heatmap, SIZE, SIZE);

        float[] out = new float[3 * plane];
        for (int i = 0; i < plane; i++) {
            int p = resized[i];
            float[] rgb = {((p >> 16) & 0xFF) / 255f, ((p >> 8) & 0xFF) / 255f, (p & 0xFF) / 255f};
            for (int c = 0; c < 3; c++) {
                float blended = BLEND_ALPHA * rgb[c] + (1f - BLEND_ALPHA) * colour[c * plane + i];
                out[c * plane + i] = (blended - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
        return out;
    }

    private static float minMax(float v, float lo, float hi) {
        if (Math.abs(hi - lo) < 1e-9f) return 0.5f;
        return clamp((v - lo) / (hi - lo), 0f, 1f);
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
