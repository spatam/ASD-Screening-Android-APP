package it.unict.dmi.asdscreening;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;

/**
 * The 300 Saliency4ASD stimuli, numbered 0001.png to 0300.png.
 *
 * The photos come from MIT1003 and cannot be redistributed, so the public APK ships without them.
 * They are read from either of two places:
 *  - assets/stimuli/, for local builds prepared with scripts/fetch_saliency4asd.py;
 *  - the app's private storage, after the user imports them (see {@link StimulusImporter}).
 */
public class StimulusRepository {

    public static final int COUNT = 300;

    private final AssetManager assets;
    private final File importedDir;
    private final boolean bundled;

    public StimulusRepository(Context context) {
        this.assets = context.getAssets();
        this.importedDir = importedDir(context);
        this.bundled = countBundled(assets) >= COUNT;
    }

    public static File importedDir(Context context) {
        return new File(context.getFilesDir(), "stimuli");
    }

    /** File name of stimulus {@code index0} (0-based), as in assets and in private storage. */
    public static String fileName(int index0) {
        return String.format(Locale.US, "%04d.png", index0 + 1);
    }

    /** True when every stimulus can be loaded. */
    public boolean isComplete() {
        if (bundled) return true;
        for (int i = 0; i < COUNT; i++) {
            if (!new File(importedDir, fileName(i)).isFile()) return false;
        }
        return true;
    }

    public Bitmap loadStimulusBitmap(int index0) throws IOException {
        if (bundled) {
            try (InputStream is = assets.open("stimuli/" + fileName(index0))) {
                return decode(BitmapFactory.decodeStream(is), index0);
            }
        }
        File file = new File(importedDir, fileName(index0));
        return decode(BitmapFactory.decodeFile(file.getPath()), index0);
    }

    private static Bitmap decode(Bitmap bitmap, int index0) throws IOException {
        if (bitmap == null) throw new IOException("Cannot decode stimulus " + fileName(index0));
        return bitmap;
    }

    private static int countBundled(AssetManager assets) {
        try {
            String[] names = assets.list("stimuli");
            return names == null ? 0 : names.length;
        } catch (IOException e) {
            return 0;
        }
    }
}
