package it.unict.dmi.asdscreening;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class FixationDetectorTest {

    private static final long PERIOD_MS = 33;

    private static void hold(List<GazeSample> out, long startMs, long durationMs, float x, float y) {
        for (long t = startMs; t < startMs + durationMs; t += PERIOD_MS) {
            out.add(new GazeSample(t, x, y, true));
        }
    }

    @Test
    public void twoFixationsInStimulusPixels() {
        List<GazeSample> samples = new ArrayList<>();
        hold(samples, 0, 495, 0.25f, 0.5f);
        hold(samples, 528, 495, 0.75f, 0.5f);

        List<Fixation> fixations = FixationDetector.detectFixations(samples, 1024, 768);

        assertEquals(2, fixations.size());
        assertEquals(256f, fixations.get(0).x, 1e-3f);
        assertEquals(384f, fixations.get(0).y, 1e-3f);
        assertEquals(768f, fixations.get(1).x, 1e-3f);
        for (Fixation f : fixations) {
            assertTrue("duration " + f.durationMs, f.durationMs >= 400 && f.durationMs <= 600);
        }
    }

    @Test
    public void blinkSplitsAFixation() {
        List<GazeSample> samples = new ArrayList<>();
        hold(samples, 0, 297, 0.5f, 0.5f);
        samples.add(new GazeSample(300, 0, 0, false));
        hold(samples, 330, 297, 0.5f, 0.5f);

        assertEquals(2, FixationDetector.detectFixations(samples, 1024, 768).size());
    }

    @Test
    public void offImageFixationsAreDropped() {
        List<GazeSample> samples = new ArrayList<>();
        hold(samples, 0, 495, 1.4f, 0.5f);
        assertTrue(FixationDetector.detectFixations(samples, 1024, 768).isEmpty());
    }
}
