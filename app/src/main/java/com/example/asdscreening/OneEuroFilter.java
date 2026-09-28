package com.example.asdscreening;

/**
 * One Euro filter (Casiez et al., CHI 2012): adaptive low-pass that removes jitter during
 * fixations while keeping latency low during saccades.
 */
public final class OneEuroFilter {

    private final double minCutoff;
    private final double beta;
    private final double dCutoff;

    private boolean initialised = false;
    private double xPrev, dxPrev;
    private long tPrevMs;

    public OneEuroFilter(double minCutoffHz, double beta, double dCutoffHz) {
        this.minCutoff = minCutoffHz;
        this.beta = beta;
        this.dCutoff = dCutoffHz;
    }

    public void reset() { initialised = false; }

    public double filter(double x, long tMs) {
        if (!initialised) {
            initialised = true;
            xPrev = x; dxPrev = 0; tPrevMs = tMs;
            return x;
        }
        double dt = Math.max(1e-3, (tMs - tPrevMs) / 1000.0);
        tPrevMs = tMs;
        double dx = (x - xPrev) / dt;
        double aD = alpha(dCutoff, dt);
        double dxHat = aD * dx + (1 - aD) * dxPrev;
        double cutoff = minCutoff + beta * Math.abs(dxHat);
        double a = alpha(cutoff, dt);
        double xHat = a * x + (1 - a) * xPrev;
        xPrev = xHat; dxPrev = dxHat;
        return xHat;
    }

    private static double alpha(double cutoff, double dt) {
        double tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }
}
