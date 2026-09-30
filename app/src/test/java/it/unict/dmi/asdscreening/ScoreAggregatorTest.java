package it.unict.dmi.asdscreening;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ScoreAggregatorTest {

    @Test
    public void sessionScoreAveragesLogits() {
        // logit(0.9) = 2.1972, logit(0.6) = 0.4055, mean 1.3013, sigmoid 0.7861.
        // Averaging probabilities would give 0.75.
        assertEquals(0.7861f, ScoreAggregator.sessionScore(Arrays.asList(0.9f, 0.6f)), 1e-4f);
    }

    @Test
    public void sessionScoreIsNeutralWithoutImages() {
        assertEquals(0.5f, ScoreAggregator.sessionScore(Collections.emptyList()), 0f);
        assertEquals(0.5f, ScoreAggregator.sessionScore(null), 0f);
    }

    @Test
    public void extremeProbabilitiesStayFinite() {
        float s = ScoreAggregator.sessionScore(Arrays.asList(0f, 1f, 1f));
        assertTrue(s > 0.5f && s < 1f);
    }

    @Test
    public void verdictWaitsForEnoughImages() {
        List<Float> scores = new ArrayList<>(Collections.nCopies(ScoreAggregator.MIN_IMAGES_FOR_VERDICT - 1, 0.9f));
        assertFalse(ScoreAggregator.hasVerdict(scores.size()));
        assertTrue(ScoreAggregator.verdictText(0.9f, scores.size()).startsWith("collecting"));
        assertTrue(ScoreAggregator.hasVerdict(ScoreAggregator.MIN_IMAGES_FOR_VERDICT));
    }

    @Test
    public void verdictUsesStudyThreshold() {
        int n = ScoreAggregator.MIN_IMAGES_FOR_VERDICT;
        assertTrue(ScoreAggregator.verdictText(ScoreAggregator.STUDY_THRESHOLD + 0.01f, n).startsWith("above"));
        assertTrue(ScoreAggregator.verdictText(ScoreAggregator.STUDY_THRESHOLD - 0.01f, n).startsWith("below"));
    }
}
