package com.example.asdscreening;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

public class StimulusRepository {
    private final AssetManager assets;

    public StimulusRepository(AssetManager assets) {
        this.assets = assets;
    }

    /**
     * Expected assets path:
     *   app/src/main/assets/stimuli/1.png ... 300.png  (0001.png and .jpg also accepted)
     *
     * */

    public Bitmap loadStimulusBitmap(int index0) throws IOException {
        int idx1 = index0 + 1;
        String[] candidates = {
                "stimuli/" + idx1 + ".png",
                String.format(Locale.US, "stimuli/%04d.png", idx1),
                "stimuli/" + idx1 + ".jpg",
                String.format(Locale.US, "stimuli/%04d.jpg", idx1)
        };
        IOException last = null;
        for (String name : candidates) {
            try (InputStream is = assets.open(name)) {
                Bitmap bmp = BitmapFactory.decodeStream(is);
                if (bmp == null) throw new IOException("Bitmap decode failed for " + name);
                return bmp;
            } catch (IOException e) {
                last = e;
            }
        }
        throw last != null ? last : new IOException("Stimulus not found: " + idx1);
    }
}
