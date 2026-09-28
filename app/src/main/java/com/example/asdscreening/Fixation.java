package com.example.asdscreening;

public class Fixation {
    public final float xScreen;       // in [0..1280] nominal space
    public final float yScreen;       // in [0..1024] nominal space
    public final float durationMs;

    public Fixation(float xScreen, float yScreen, float durationMs) {
        this.xScreen = xScreen;
        this.yScreen = yScreen;
        this.durationMs = durationMs;
    }
}
