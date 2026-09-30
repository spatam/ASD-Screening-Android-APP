package it.unict.dmi.asdscreening;

/**
 * One gaze sample, in coordinates NORMALISED TO THE DISPLAYED STIMULUS IMAGE:
 * (0,0) = top-left corner of the picture actually drawn on screen, (1,1) = bottom-right.
 * Values slightly outside [0,1] mean the subject looked outside the picture.
 */
public class GazeSample {
    public final long tMs;       // SystemClock.elapsedRealtime()
    public final float xNorm;
    public final float yNorm;
    public final boolean valid;  // false = no face / blink / pupil not found

    public GazeSample(long tMs, float xNorm, float yNorm, boolean valid) {
        this.tMs = tMs;
        this.xNorm = xNorm;
        this.yNorm = yNorm;
        this.valid = valid;
    }
}
