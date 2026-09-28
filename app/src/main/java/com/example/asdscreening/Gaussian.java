package com.example.asdscreening;

public class Gaussian {

    /**
     * Separable gaussian blur on a single-channel image stored as 1D array row-major.
     * sigma in pixels.
     */
    public static float[] gaussianBlur(float[] src, int h, int w, float sigma) {
        if (sigma <= 0.0f) return src;

        int radius = Math.max(1, (int) Math.ceil(3.0 * sigma));
        float[] kernel = gaussianKernel1D(radius, sigma);

        float[] tmp = new float[h * w];
        float[] dst = new float[h * w];

        // horizontal
        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++) {
                float sum = 0f;
                for (int k = -radius; k <= radius; k++) {
                    int xx = clampInt(x + k, 0, w - 1);
                    sum += src[row + xx] * kernel[k + radius];
                }
                tmp[row + x] = sum;
            }
        }

        // vertical
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                float sum = 0f;
                for (int k = -radius; k <= radius; k++) {
                    int yy = clampInt(y + k, 0, h - 1);
                    sum += tmp[yy * w + x] * kernel[k + radius];
                }
                dst[y * w + x] = sum;
            }
        }

        return dst;
    }

    private static float[] gaussianKernel1D(int radius, float sigma) {
        int size = 2 * radius + 1;
        float[] k = new float[size];
        double s2 = 2.0 * sigma * sigma;
        double sum = 0.0;
        for (int i = -radius; i <= radius; i++) {
            double v = Math.exp(-(i * i) / s2);
            k[i + radius] = (float) v;
            sum += v;
        }
        // normalize
        if (sum > 0) {
            for (int i = 0; i < size; i++) k[i] /= (float) sum;
        }
        return k;
    }

    private static int clampInt(int v, int lo, int hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }
}
