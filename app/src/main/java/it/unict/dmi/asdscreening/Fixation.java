package it.unict.dmi.asdscreening;

/**
 * One fixation in stimulus pixel coordinates: (0,0) is the top-left corner of the picture and
 * (width, height) its bottom-right corner, in the picture's own resolution. The Saliency4ASD
 * scanpath files use the same convention, so the app feeds the models what they saw in training.
 */
public class Fixation {
    public final float x;
    public final float y;
    public final float durationMs;

    public Fixation(float x, float y, float durationMs) {
        this.x = x;
        this.y = y;
        this.durationMs = durationMs;
    }
}
