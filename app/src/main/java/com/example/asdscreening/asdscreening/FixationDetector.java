package com.example.asdscreening;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple dispersion-based fixation detection.
 * Input: gaze samples (normalized), ~30-60Hz.
 * Output: list of fixations (x_screen,y_screen,duration_ms) in nominal 1280x1024 space.
 *
 * This is intentionally conservative and deterministic. If you already have a fixation detector,
 * replace this class and feed its fixations directly.
 */
public class FixationDetector {

    // parameters (tune if you move to real gaze)
    private static final long MIN_FIX_MS = 80;         // minimum fixation duration
    private static final float DISPERSION_THRESH = 0.03f; // in normalized units

    public static List<Fixation> detectFixations(List<CameraGazeTracker.GazeSample> samples) {
        List<Fixation> out = new ArrayList<>();
        if (samples == null || samples.isEmpty()) return out;

        int i = 0;
        while (i < samples.size()) {
            int j = i;

            float minX = samples.get(i).xNorm, maxX = samples.get(i).xNorm;
            float minY = samples.get(i).yNorm, maxY = samples.get(i).yNorm;

            while (j < samples.size()) {
                float x = samples.get(j).xNorm;
                float y = samples.get(j).yNorm;
                minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                minY = Math.min(minY, y); maxY = Math.max(maxY, y);

                float dispersion = (maxX - minX) + (maxY - minY);
                if (dispersion > DISPERSION_THRESH && j > i) {
                    break;
                }
                j++;
            }

            long tStart = samples.get(i).tMs;
            long tEnd = samples.get(j - 1).tMs;
            long dur = Math.max(0, tEnd - tStart);

            if (dur >= MIN_FIX_MS) {
                // average point
                float sumX = 0, sumY = 0;
                int count = 0;
                for (int k = i; k < j; k++) {
                    sumX += samples.get(k).xNorm;
                    sumY += samples.get(k).yNorm;
                    count++;
                }
                float xAvg = (count > 0) ? (sumX / count) : samples.get(i).xNorm;
                float yAvg = (count > 0) ? (sumY / count) : samples.get(i).yNorm;

                // map to nominal 1280x1024 coordinate system
                float xScreen = xAvg * 1280.0f;
                float yScreen = yAvg * 1024.0f;

                out.add(new Fixation(xScreen, yScreen, (float) dur));
            }

            i = Math.max(j, i + 1);
        }

        return out;
    }
}
