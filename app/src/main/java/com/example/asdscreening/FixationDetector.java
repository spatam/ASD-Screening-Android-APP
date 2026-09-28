package com.example.asdscreening;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Dispersion-threshold fixation identification (I-DT, Salvucci & Goldberg 2000).
 *
 * Input : gaze samples normalised to the displayed stimulus image (see {@link GazeSample}).
 * Output: fixations in the nominal 1280x1024 space expected by {@link Preprocessing}.
 *
 * Differences from the previous version:
 *  - invalid samples (blink / no face) and time gaps split the stream: a fixation never
 *    spans a blink;
 *  - classic I-DT window growth (initial window = MIN_FIX_MS, slide by one sample when the
 *    window is not a fixation) instead of skipping whole blocks;
 *  - duration includes the last sample period;
 *  - fixations whose centroid lies outside the picture are discarded (the training data
 *    only contains on-image fixations);
 *  - thresholds are expressed in image-normalised units, not camera-frame units.
 */
public class FixationDetector {

    /** Minimum fixation duration. */
    public static long MIN_FIX_MS = 100;
    /** (maxX-minX)+(maxY-minY) in image-normalised units (~1.5-2 deg on a phone at 35 cm). */
    public static float DISPERSION_THRESH = 0.12f;
    /** A gap between valid samples larger than this splits fixations (blink/tracking loss). */
    public static long MAX_GAP_MS = 150;
    /** Tolerance for fixations just outside the picture border (clamped to the border). */
    public static float OUTSIDE_TOLERANCE = 0.05f;

    public static List<Fixation> detectFixations(List<GazeSample> samples) {
        List<Fixation> out = new ArrayList<>();
        if (samples == null || samples.isEmpty()) return out;

        long period = medianPeriod(samples);

        // split into runs of valid, temporally contiguous samples
        List<GazeSample> run = new ArrayList<>();
        GazeSample prev = null;
        for (GazeSample s : samples) {
            if (!s.valid) {
                processRun(run, period, out);
                run.clear();
                prev = null;
                continue;
            }
            if (prev != null && s.tMs - prev.tMs > MAX_GAP_MS) {
                processRun(run, period, out);
                run.clear();
            }
            run.add(s);
            prev = s;
        }
        processRun(run, period, out);
        return out;
    }

    private static void processRun(List<GazeSample> r, long period, List<Fixation> out) {
        int n = r.size();
        int i = 0;
        while (i < n) {
            // initial window covering MIN_FIX_MS
            int j = i;
            while (j < n && r.get(j).tMs - r.get(i).tMs + period < MIN_FIX_MS) j++;
            if (j >= n) break;

            if (dispersion(r, i, j) <= DISPERSION_THRESH) {
                while (j + 1 < n && dispersion(r, i, j + 1) <= DISPERSION_THRESH) j++;
                emit(r, i, j, period, out);
                i = j + 1;
            } else {
                i++;
            }
        }
    }

    private static void emit(List<GazeSample> r, int i, int j, long period, List<Fixation> out) {
        double sx = 0, sy = 0;
        for (int k = i; k <= j; k++) { sx += r.get(k).xNorm; sy += r.get(k).yNorm; }
        int cnt = j - i + 1;
        float x = (float) (sx / cnt), y = (float) (sy / cnt);
        if (x < -OUTSIDE_TOLERANCE || x > 1 + OUTSIDE_TOLERANCE
                || y < -OUTSIDE_TOLERANCE || y > 1 + OUTSIDE_TOLERANCE) {
            return; // off-image fixation
        }
        x = Math.max(0f, Math.min(1f, x));
        y = Math.max(0f, Math.min(1f, y));
        float dur = (float) (r.get(j).tMs - r.get(i).tMs + period);
        out.add(new Fixation(x * 1280.0f, y * 1024.0f, dur));
    }

    private static float dispersion(List<GazeSample> r, int i, int j) {
        float minX = Float.MAX_VALUE, maxX = -Float.MAX_VALUE;
        float minY = Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (int k = i; k <= j; k++) {
            GazeSample s = r.get(k);
            if (s.xNorm < minX) minX = s.xNorm;
            if (s.xNorm > maxX) maxX = s.xNorm;
            if (s.yNorm < minY) minY = s.yNorm;
            if (s.yNorm > maxY) maxY = s.yNorm;
        }
        return (maxX - minX) + (maxY - minY);
    }

    private static long medianPeriod(List<GazeSample> s) {
        if (s.size() < 2) return 33;
        long[] d = new long[s.size() - 1];
        for (int k = 1; k < s.size(); k++) d[k - 1] = s.get(k).tMs - s.get(k - 1).tMs;
        Arrays.sort(d);
        long m = d[d.length / 2];
        return Math.max(1, Math.min(m, 200));
    }
}
