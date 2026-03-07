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
     *   app/src/main/assets/stimuli/0001.png ... 0300.png
     *
     * */

    public Bitmap loadStimulusBitmap(int index0) throws IOException {
        int idx1 = index0 + 1;
        String name = "stimuli/" + idx1 + ".png";

        try (InputStream is = assets.open(name)) {
            Bitmap bmp = BitmapFactory.decodeStream(is);
            if (bmp == null) throw new IOException("Bitmap decode failed for " + name);
            return bmp;
        }
    }
}
