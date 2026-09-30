package it.unict.dmi.asdscreening;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.BeforeClass;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

/**
 * Checks the Java preprocessing against asd_gaze/preprocessing.py.
 * Regenerate the reference file with {@code python scripts/make_parity_golden.py}.
 */
public class PreprocessingParityTest {

    private static final List<int[]> IMAGES = new ArrayList<>();          // {width, height}
    private static final List<String> RESIZE_SHA256 = new ArrayList<>();
    private static final List<Fixation> FIXATIONS = new ArrayList<>();
    private static float[] gazeSeq;
    private static double visualSum;
    private static int[] sampleIndex;
    private static float[] sampleValue;

    @BeforeClass
    public static void loadGolden() throws Exception {
        InputStream in = PreprocessingParityTest.class.getClassLoader().getResourceAsStream("parity_golden.txt");
        assertNotNull("parity_golden.txt missing from test resources", in);
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.US_ASCII))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("#") || line.isEmpty()) continue;
                String[] t = line.trim().split("\\s+");
                switch (t[0]) {
                    case "image":
                        IMAGES.add(new int[]{Integer.parseInt(t[1]), Integer.parseInt(t[2])});
                        RESIZE_SHA256.add(t[3]);
                        break;
                    case "fixations":
                        int n = Integer.parseInt(t[1]);
                        for (int i = 0; i < n; i++) {
                            String[] f = reader.readLine().trim().split("\\s+");
                            FIXATIONS.add(new Fixation(Float.parseFloat(f[0]), Float.parseFloat(f[1]), Float.parseFloat(f[2])));
                        }
                        break;
                    case "gaze_seq":
                        gazeSeq = new float[t.length - 1];
                        for (int i = 1; i < t.length; i++) gazeSeq[i - 1] = Float.parseFloat(t[i]);
                        break;
                    case "visual_sum":
                        visualSum = Double.parseDouble(t[1]);
                        break;
                    case "visual_samples":
                        int m = Integer.parseInt(t[1]);
                        sampleIndex = new int[m];
                        sampleValue = new float[m];
                        for (int i = 0; i < m; i++) {
                            String[] s = reader.readLine().trim().split("\\s+");
                            sampleIndex[i] = Integer.parseInt(s[0]);
                            sampleValue[i] = Float.parseFloat(s[1]);
                        }
                        break;
                    default:
                        throw new IllegalStateException("Unknown golden entry: " + t[0]);
                }
            }
        }
    }

    /** Same formula as procedural_image() in scripts/make_parity_golden.py. */
    static int[] proceduralImage(int width, int height) {
        int[] argb = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (x * 7 + y * 3) & 255;
                int g = ((x ^ y) * 5) & 255;
                int b = ((x * x + y * 13) >> 4) & 255;
                argb[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return argb;
    }

    @Test
    public void resizeMatchesPillowBitForBit() throws Exception {
        for (int k = 0; k < IMAGES.size(); k++) {
            int w = IMAGES.get(k)[0], h = IMAGES.get(k)[1];
            int[] out = PilResize.bilinear(proceduralImage(w, h), w, h, Preprocessing.SIZE, Preprocessing.SIZE);
            byte[] rgb = new byte[out.length * 3];
            for (int i = 0; i < out.length; i++) {
                rgb[3 * i] = (byte) (out[i] >> 16);
                rgb[3 * i + 1] = (byte) (out[i] >> 8);
                rgb[3 * i + 2] = (byte) out[i];
            }
            assertEquals("resize of the " + w + "x" + h + " picture", RESIZE_SHA256.get(k), sha256(rgb));
        }
    }

    @Test
    public void gazeSequenceMatchesPython() {
        float[] seq = Preprocessing.buildGazeSeq(FIXATIONS);
        assertEquals(gazeSeq.length, seq.length);
        for (int i = 0; i < seq.length; i++) {
            assertEquals("gaze_seq[" + i + "]", gazeSeq[i], seq[i], 1e-6f);
        }
    }

    @Test
    public void visualTensorMatchesPython() {
        float[] visual = Preprocessing.buildStimulusImageTensor(proceduralImage(1024, 768), 1024, 768, FIXATIONS);
        assertEquals(3 * Preprocessing.SIZE * Preprocessing.SIZE, visual.length);
        for (int i = 0; i < sampleIndex.length; i++) {
            assertEquals("visual[" + sampleIndex[i] + "]", sampleValue[i], visual[sampleIndex[i]], 2e-4f);
        }
        double sum = 0;
        for (float v : visual) sum += v;
        assertEquals(visualSum, sum, 1.0);
    }

    @Test
    public void emptyFixationsGiveZeroSequence() {
        float[] seq = Preprocessing.buildGazeSeq(new ArrayList<>());
        for (float v : seq) assertEquals(0f, v, 0f);
    }

    private static String sha256(byte[] data) throws Exception {
        byte[] d = MessageDigest.getInstance("SHA-256").digest(data);
        StringBuilder sb = new StringBuilder();
        for (byte b : d) sb.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        return sb.toString();
    }
}
