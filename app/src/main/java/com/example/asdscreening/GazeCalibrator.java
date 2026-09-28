package com.example.asdscreening;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Per-subject gaze calibration: maps eye features to SCREEN pixels with a regularised
 * second-order polynomial (ridge regression), the standard approach for camera-based
 * gaze estimation.
 *
 * Input feature vector f (length {@link #NF}):
 *   f[0] = ex    horizontal iris offset (mean of both eyes, normalised by eye width)
 *   f[1] = ey    vertical iris offset
 *   f[2] = yaw   head yaw proxy   (nose tip vs eye midpoint, / interocular distance)
 *   f[3] = pitch head pitch proxy
 *
 * Expanded regressors: [1, ex, ey, ex*ey, ex^2, ey^2, yaw, pitch], standardised.
 * Regressors that do not vary during calibration (e.g. a perfectly still head) are
 * dropped automatically instead of being extrapolated.
 *
 * Thread-safety: all public methods are synchronized.
 */
public final class GazeCalibrator {

    public static final int NF = 4;
    private static final int NP = 7;               // expanded regressors (without bias)
    private static final double[] MIN_STD = {1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 0.01, 0.01};

    public static final class FitResult {
        public final boolean ok;
        public final String message;
        public final int pointsUsed;
        public final int samplesUsed;
        /** Mean error of per-point median prediction, in screen pixels. */
        public final float meanErrorPx;
        public final float maxErrorPx;

        FitResult(boolean ok, String message, int pointsUsed, int samplesUsed,
                  float meanErrorPx, float maxErrorPx) {
            this.ok = ok; this.message = message;
            this.pointsUsed = pointsUsed; this.samplesUsed = samplesUsed;
            this.meanErrorPx = meanErrorPx; this.maxErrorPx = maxErrorPx;
        }
    }

    private static final class Sample {
        final int point; final float tx, ty; final float[] f;
        Sample(int point, float tx, float ty, float[] f) {
            this.point = point; this.tx = tx; this.ty = ty; this.f = f;
        }
    }

    private final List<Sample> samples = new ArrayList<>();

    // fitted model
    private boolean fitted = false;
    private final double[] mean = new double[NP];
    private final double[] scale = new double[NP];   // 1/std, or 0 if dropped
    private double[] betaX, betaY;                   // length NP+1 (bias first)

    public int minSamplesPerPoint = 4;
    public int minPoints = 6;
    public double lambda = 0.003;

    public synchronized void clearSamples() { samples.clear(); }

    public synchronized void addSample(int pointId, float targetX, float targetY, float[] features) {
        samples.add(new Sample(pointId, targetX, targetY, features.clone()));
    }

    public synchronized int sampleCount() { return samples.size(); }

    public synchronized boolean isFitted() { return fitted; }

    public synchronized FitResult fit() {
        // ---- group by point & reject outliers (saccades, blinks mis-detected, etc.)
        int maxId = -1;
        for (Sample s : samples) maxId = Math.max(maxId, s.point);
        List<Sample> kept = new ArrayList<>();
        int pointsUsed = 0;
        List<float[]> pointTargets = new ArrayList<>();
        List<List<Sample>> perPoint = new ArrayList<>();
        for (int p = 0; p <= maxId; p++) {
            List<Sample> ps = new ArrayList<>();
            for (Sample s : samples) if (s.point == p) ps.add(s);
            if (ps.size() < minSamplesPerPoint) continue;
            float medX = median(ps, 0), medY = median(ps, 1);
            float madX = mad(ps, 0, medX), madY = mad(ps, 1, medY);
            float thX = 3.5f * Math.max(1.4826f * madX, 0.004f);
            float thY = 3.5f * Math.max(1.4826f * madY, 0.004f);
            List<Sample> good = new ArrayList<>();
            for (Sample s : ps) {
                if (Math.abs(s.f[0] - medX) <= thX && Math.abs(s.f[1] - medY) <= thY) good.add(s);
            }
            if (good.size() < minSamplesPerPoint) continue;
            kept.addAll(good);
            perPoint.add(good);
            pointTargets.add(new float[]{good.get(0).tx, good.get(0).ty});
            pointsUsed++;
        }
        if (pointsUsed < minPoints) {
            fitted = false;
            return new FitResult(false, "Not enough valid calibration points (" + pointsUsed + ")",
                    pointsUsed, kept.size(), Float.NaN, Float.NaN);
        }

        // ---- standardisation
        int n = kept.size();
        double[][] phi = new double[n][];
        for (int i = 0; i < n; i++) phi[i] = expand(kept.get(i).f);
        for (int k = 0; k < NP; k++) {
            double m = 0;
            for (int i = 0; i < n; i++) m += phi[i][k];
            m /= n;
            double v = 0;
            for (int i = 0; i < n; i++) { double d = phi[i][k] - m; v += d * d; }
            double sd = Math.sqrt(v / Math.max(1, n - 1));
            mean[k] = m;
            scale[k] = (sd < MIN_STD[k]) ? 0.0 : 1.0 / sd;
        }

        // ---- ridge normal equations (bias not penalised)
        int d = NP + 1;
        double[][] A = new double[d][d];
        double[] bx = new double[d], by = new double[d];
        double[] z = new double[d];
        for (int i = 0; i < n; i++) {
            standardise(phi[i], z);
            Sample s = kept.get(i);
            for (int r = 0; r < d; r++) {
                bx[r] += z[r] * s.tx;
                by[r] += z[r] * s.ty;
                for (int c = 0; c < d; c++) A[r][c] += z[r] * z[c];
            }
        }
        for (int r = 1; r < d; r++) {
            A[r][r] += lambda * n;
            if (scale[r - 1] == 0.0) A[r][r] += 1.0;  // dropped regressor: keep system regular
        }
        double[] solX = solve(copy(A), bx.clone());
        double[] solY = solve(copy(A), by.clone());
        if (solX == null || solY == null) {
            fitted = false;
            return new FitResult(false, "Singular calibration system", pointsUsed, n, Float.NaN, Float.NaN);
        }
        betaX = solX; betaY = solY; fitted = true;

        // ---- error on per-point median predictions
        double sumErr = 0, maxErr = 0;
        for (int p = 0; p < perPoint.size(); p++) {
            List<Sample> ps = perPoint.get(p);
            float[] px = new float[ps.size()], py = new float[ps.size()];
            for (int i = 0; i < ps.size(); i++) {
                float[] pr = predictUnsafe(ps.get(i).f);
                px[i] = pr[0]; py[i] = pr[1];
            }
            Arrays.sort(px); Arrays.sort(py);
            float mx = px[px.length / 2], my = py[py.length / 2];
            double e = Math.hypot(mx - pointTargets.get(p)[0], my - pointTargets.get(p)[1]);
            sumErr += e; maxErr = Math.max(maxErr, e);
        }
        return new FitResult(true, "OK", pointsUsed, n,
                (float) (sumErr / perPoint.size()), (float) maxErr);
    }

    /** Returns {screenX, screenY} or null when not calibrated. */
    public synchronized float[] predict(float[] f) {
        if (!fitted) return null;
        return predictUnsafe(f);
    }

    // ------------------------------------------------------------------------------------

    private float[] predictUnsafe(float[] f) {
        double[] z = new double[NP + 1];
        standardise(expand(f), z);
        double x = 0, y = 0;
        for (int k = 0; k <= NP; k++) { x += betaX[k] * z[k]; y += betaY[k] * z[k]; }
        return new float[]{(float) x, (float) y};
    }

    private static double[] expand(float[] f) {
        double ex = f[0], ey = f[1];
        return new double[]{ex, ey, ex * ey, ex * ex, ey * ey, f[2], f[3]};
    }

    private void standardise(double[] phi, double[] z) {
        z[0] = 1.0;
        for (int k = 0; k < NP; k++) z[k + 1] = (phi[k] - mean[k]) * scale[k];
    }

    private static float median(List<Sample> ps, int k) {
        float[] v = new float[ps.size()];
        for (int i = 0; i < v.length; i++) v[i] = ps.get(i).f[k];
        Arrays.sort(v);
        return v[v.length / 2];
    }

    private static float mad(List<Sample> ps, int k, float med) {
        float[] v = new float[ps.size()];
        for (int i = 0; i < v.length; i++) v[i] = Math.abs(ps.get(i).f[k] - med);
        Arrays.sort(v);
        return v[v.length / 2];
    }

    private static double[][] copy(double[][] a) {
        double[][] c = new double[a.length][];
        for (int i = 0; i < a.length; i++) c[i] = a[i].clone();
        return c;
    }

    /** Gaussian elimination with partial pivoting. */
    private static double[] solve(double[][] A, double[] b) {
        int n = b.length;
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
            if (Math.abs(A[piv][col]) < 1e-12) return null;
            double[] t = A[col]; A[col] = A[piv]; A[piv] = t;
            double tb = b[col]; b[col] = b[piv]; b[piv] = tb;
            for (int r = col + 1; r < n; r++) {
                double fct = A[r][col] / A[col][col];
                if (fct == 0) continue;
                for (int c = col; c < n; c++) A[r][c] -= fct * A[col][c];
                b[r] -= fct * b[col];
            }
        }
        double[] x = new double[n];
        for (int r = n - 1; r >= 0; r--) {
            double s = b[r];
            for (int c = r + 1; c < n; c++) s -= A[r][c] * x[c];
            x[r] = s / A[r][r];
        }
        return x;
    }
}
