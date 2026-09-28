package com.example.asdscreening;

import java.util.List;

public class ScoreAggregator {

    public static float sigmoid(float x) {
        // stable-ish sigmoid
        if (x >= 0) {
            double z = Math.exp(-x);
            return (float) (1.0 / (1.0 + z));
        } else {
            double z = Math.exp(x);
            return (float) (z / (1.0 + z));
        }
    }

    public static float runningMean(List<Float> vals) {
        if (vals == null || vals.isEmpty()) return 0.5f;
        double s = 0.0;
        for (Float v : vals) s += v;
        return (float) (s / vals.size());
    }

    public static String verdictText(float pAsd) {
        if (pAsd >= 0.85f) return "High ASD likelihood";
        if (pAsd <= 0.15f) return "High TD likelihood";
        return "Uncertain";
    }
}
