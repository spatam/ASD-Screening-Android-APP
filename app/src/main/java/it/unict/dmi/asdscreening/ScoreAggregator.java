package it.unict.dmi.asdscreening;

import java.util.List;
import java.util.Locale;

/**
 * Pools per-image predictions into one session score, as in the paper's deployment protocol.
 *
 * The session score is the sigmoid of the mean per-image logit. Averaging in logit space is the
 * convention behind the paper's per-subject AUC of 0.959 (probability-space averaging gives a
 * different number, 0.964, on the same predictions).
 */
public final class ScoreAggregator {

    /**
     * Decision threshold on the session score. It sits midway between the two children on either
     * side of the Youden cut of the 28 Saliency4ASD children (out-of-fold deployment predictions,
     * 0.4939 and 0.4905). It is an operating point of the study, not a clinical cut-off.
     */
    public static final float STUDY_THRESHOLD = 0.4922f;

    /** The paper keeps a per-subject AUC of 0.947 with 40 stimuli (under three minutes). */
    public static final int MIN_IMAGES_FOR_VERDICT = 40;

    private static final float EPS = 1e-6f;

    private ScoreAggregator() {}

    public static float sigmoid(float x) {
        if (x >= 0) {
            double z = Math.exp(-x);
            return (float) (1.0 / (1.0 + z));
        }
        double z = Math.exp(x);
        return (float) (z / (1.0 + z));
    }

    public static float logit(float p) {
        double q = Math.max(EPS, Math.min(1.0 - EPS, p));
        return (float) Math.log(q / (1.0 - q));
    }

    /** Sigmoid of the mean logit; 0.5 when nothing has been scored yet. */
    public static float sessionScore(List<Float> perImageProbabilities) {
        if (perImageProbabilities == null || perImageProbabilities.isEmpty()) return 0.5f;
        double sum = 0.0;
        for (Float p : perImageProbabilities) sum += logit(p);
        return sigmoid((float) (sum / perImageProbabilities.size()));
    }

    public static boolean hasVerdict(int scoredImages) {
        return scoredImages >= MIN_IMAGES_FOR_VERDICT;
    }

    public static String verdictText(float sessionScore, int scoredImages) {
        if (!hasVerdict(scoredImages)) {
            return String.format(Locale.US, "collecting (%d/%d images)", scoredImages, MIN_IMAGES_FOR_VERDICT);
        }
        return sessionScore >= STUDY_THRESHOLD
                ? "above study threshold (gaze closer to the ASD group)"
                : "below study threshold (gaze closer to the TD group)";
    }
}
