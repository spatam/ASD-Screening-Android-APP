package it.unict.dmi.asdscreening;

/**
 * Bilinear resize that reproduces Pillow's {@code Image.resize(size, Image.BILINEAR)} bit for bit
 * on 8-bit RGB images.
 *
 * The training pipeline resizes every stimulus with Pillow. When Pillow shrinks an image, its
 * bilinear filter widens with the scale factor and averages every source pixel under the output
 * pixel. {@code Bitmap.createScaledBitmap} samples only the four nearest pixels, so a 1024 to 224
 * reduction gives a different, aliased picture and moves the model's predictions (mean change of
 * 0.05 in p(ASD) on Saliency4ASD records). This class ports Pillow's ImagingResample
 * (src/libImaging/Resample.c): fixed-point weights, a horizontal pass rounded to 8 bits, then a
 * vertical pass.
 */
public final class PilResize {

    /** Same as Pillow: 32 bits minus 8 bits of pixel minus 2 guard bits. */
    private static final int PRECISION_BITS = 32 - 8 - 2;

    private PilResize() {}

    /**
     * @param argb pixels in {@code Bitmap#getPixels} layout (row-major, 0xAARRGGBB)
     * @return resized pixels as 0xFFRRGGBB, row-major, {@code outW * outH} entries
     */
    public static int[] bilinear(int[] argb, int inW, int inH, int outW, int outH) {
        if (argb.length < inW * inH) throw new IllegalArgumentException("pixel buffer too small");
        Coefficients horizontal = Coefficients.compute(inW, outW);
        Coefficients vertical = Coefficients.compute(inH, outH);

        // Like Pillow, run the horizontal pass only on the rows the vertical pass reads.
        int firstRow = vertical.bounds[0];
        int lastRow = vertical.bounds[(outH - 1) * 2] + vertical.bounds[(outH - 1) * 2 + 1];
        int rows = lastRow - firstRow;

        int[] temp = new int[rows * outW * 3];
        for (int y = 0; y < rows; y++) {
            int srcRow = (y + firstRow) * inW;
            for (int xx = 0; xx < outW; xx++) {
                int xmin = horizontal.bounds[xx * 2];
                int xmax = horizontal.bounds[xx * 2 + 1];
                int k = xx * horizontal.size;
                int r = 1 << (PRECISION_BITS - 1);
                int g = r;
                int b = r;
                for (int x = 0; x < xmax; x++) {
                    int p = argb[srcRow + xmin + x];
                    int w = horizontal.weights[k + x];
                    r += ((p >> 16) & 0xFF) * w;
                    g += ((p >> 8) & 0xFF) * w;
                    b += (p & 0xFF) * w;
                }
                int o = (y * outW + xx) * 3;
                temp[o] = clip8(r);
                temp[o + 1] = clip8(g);
                temp[o + 2] = clip8(b);
            }
        }

        int[] out = new int[outW * outH];
        for (int yy = 0; yy < outH; yy++) {
            int ymin = vertical.bounds[yy * 2] - firstRow;
            int ymax = vertical.bounds[yy * 2 + 1];
            int k = yy * vertical.size;
            for (int xx = 0; xx < outW; xx++) {
                int r = 1 << (PRECISION_BITS - 1);
                int g = r;
                int b = r;
                for (int y = 0; y < ymax; y++) {
                    int o = ((y + ymin) * outW + xx) * 3;
                    int w = vertical.weights[k + y];
                    r += temp[o] * w;
                    g += temp[o + 1] * w;
                    b += temp[o + 2] * w;
                }
                out[yy * outW + xx] = 0xFF000000 | (clip8(r) << 16) | (clip8(g) << 8) | clip8(b);
            }
        }
        return out;
    }

    private static int clip8(int value) {
        if (value >= (1 << PRECISION_BITS << 8)) return 255;
        if (value <= 0) return 0;
        return value >> PRECISION_BITS;
    }

    private static double bilinearFilter(double x) {
        if (x < 0.0) x = -x;
        return x < 1.0 ? 1.0 - x : 0.0;
    }

    /** Per-output-pixel source ranges and fixed-point weights for one axis (precompute_coeffs). */
    private static final class Coefficients {
        final int size;
        final int[] bounds;
        final int[] weights;

        private Coefficients(int size, int[] bounds, int[] weights) {
            this.size = size;
            this.bounds = bounds;
            this.weights = weights;
        }

        static Coefficients compute(int inSize, int outSize) {
            double scale = (double) inSize / outSize;
            double filterScale = Math.max(scale, 1.0);
            double support = 1.0 * filterScale;
            int size = (int) Math.ceil(support) * 2 + 1;

            int[] bounds = new int[outSize * 2];
            int[] weights = new int[outSize * size];
            double[] k = new double[size];
            for (int xx = 0; xx < outSize; xx++) {
                double center = (xx + 0.5) * scale;
                double ss = 1.0 / filterScale;
                int xmin = (int) (center - support + 0.5);
                if (xmin < 0) xmin = 0;
                int xmax = (int) (center + support + 0.5);
                if (xmax > inSize) xmax = inSize;
                xmax -= xmin;

                double total = 0.0;
                for (int x = 0; x < xmax; x++) {
                    k[x] = bilinearFilter((x + xmin - center + 0.5) * ss);
                    total += k[x];
                }
                for (int x = 0; x < xmax; x++) {
                    double w = total != 0.0 ? k[x] / total : k[x];
                    double fixed = w * (1 << PRECISION_BITS);
                    weights[xx * size + x] = (int) (w < 0 ? -0.5 + fixed : 0.5 + fixed);
                }
                bounds[xx * 2] = xmin;
                bounds[xx * 2 + 1] = xmax;
            }
            return new Coefficients(size, bounds, weights);
        }
    }
}
