package com.example.asdscreening;

import android.content.Context;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.Result;

/**
 * Loads 5 ONNX student models from assets/models/student_fold{1..5}.onnx
 *
 * Inputs:
 *  - gaze_seq: (1,25,3) float32
 *  - stimulus_image: (1,3,224,224) float32
 * Output:
 *  - logit: (1,) float32
 */
public class OnnxEnsembleRunner {

    private static final String TAG = "OnnxEnsembleRunner";

    private static final String[] MODEL_ASSETS = new String[] {
            "models/student_fold1.onnx",
            "models/student_fold2.onnx",
            "models/student_fold3.onnx",
            "models/student_fold4.onnx",
            "models/student_fold5.onnx"
    };

    private final OrtEnvironment env;
    private final SessionOptions opts;
    private final List<OrtSession> sessions = new ArrayList<>();

    private final String inputGazeName = "gaze_seq";
    private final String inputImageName = "stimulus_image";
    private final String outputName = "logit";

    /**
     * NNAPI is OFF by default: the NNAPI EP may execute in relaxed FP16 and on several SoCs
     * silently degrades distilled models (outputs collapse to a near-constant logit). It is
     * also deprecated from Android 15. Enable only after checking that CPU and NNAPI logits
     * match on the same inputs.
     */
    public static boolean USE_NNAPI = false;

    public OnnxEnsembleRunner(Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();
        opts = new SessionOptions();
        if (USE_NNAPI) {
            try { opts.addNnapi(); } catch (Throwable t) { Log.w(TAG, "NNAPI not available", t); }
        }
        opts.setIntraOpNumThreads(2);

        for (String asset : MODEL_ASSETS) {
            byte[] bytes = readAllBytes(context, asset);
            OrtSession s = env.createSession(bytes, opts);
            if (!s.getInputNames().contains(inputGazeName) || !s.getInputNames().contains(inputImageName)) {
                throw new IllegalStateException(asset + ": unexpected input names " + s.getInputNames());
            }
            Log.i(TAG, asset + " inputs=" + s.getInputNames() + " outputs=" + s.getOutputNames());
            sessions.add(s);
        }
    }

    public float runEnsembleMeanLogit(float[] gazeSeq, float[] stimulusImage) throws Exception {
        // Shapes
        long[] gazeShape = new long[]{1, 25, 3};
        long[] imgShape = new long[]{1, 3, 224, 224};

        // Create tensors once per run
        try (OnnxTensor gazeT = OnnxTensor.createTensor(env, FloatBuffer.wrap(gazeSeq), gazeShape);
             OnnxTensor imgT  = OnnxTensor.createTensor(env, FloatBuffer.wrap(stimulusImage), imgShape)) {

            double sum = 0.0;
            StringBuilder dbg = new StringBuilder();

            for (OrtSession sess : sessions) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put(inputGazeName, gazeT);
                inputs.put(inputImageName, imgT);

                try (Result res = sess.run(inputs)) {
                    Object v = res.get(0).getValue();
                    float logit = extractSingleFloat(v);
                    sum += logit;
                    dbg.append(String.format(java.util.Locale.US, "%.3f ", logit));
                }
            }
            Log.d(TAG, "fold logits: " + dbg);
            return (float) (sum / sessions.size());
        }
    }

    private static float extractSingleFloat(Object v) {
        // ORT can return float[] or float[][] etc depending on model
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            return a[0];
        } else if (v instanceof FloatBuffer) {
            FloatBuffer fb = (FloatBuffer) v;
            return fb.get(0);
        } else if (v instanceof float[][]) {
            return ((float[][]) v)[0][0];
        } else if (v instanceof float[][][]) {
            return ((float[][][]) v)[0][0][0];
        } else if (v instanceof float[][][][]) {
            return ((float[][][][]) v)[0][0][0][0];
        }
        // Fallback
        throw new IllegalArgumentException("Unexpected output type: " + v.getClass());
    }

    private static byte[] readAllBytes(Context ctx, String assetPath) throws IOException {
        try (InputStream is = ctx.getAssets().open(assetPath);
             ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int r;
            while ((r = is.read(buf)) != -1) bos.write(buf, 0, r);
            return bos.toByteArray();
        }
    }

    public void close() {
        for (OrtSession s : sessions) {
            try { s.close(); } catch (Exception ignored) {}
        }
        try { opts.close(); } catch (Exception ignored) {}
        // env is singleton; no close needed
    }
}
