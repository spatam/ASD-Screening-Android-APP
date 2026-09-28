package com.example.asdscreening;

import android.graphics.Bitmap;

import java.util.List;

/**
 * Implements your exact preprocessing:
 *  - gaze_seq: (1,25,3) from fixations with local min-max + duration log1p/10, pad/truncate.
 *  - stimulus_image: blended+normalized image (1,3,224,224):
 *      img_rgb resized -> [0,1]
 *      hmap from fixations (duration weight log1p), gaussian sigma=10, normalized
 *      jet colormap -> heatmap_rgb [0,1]
 *      blended = 0.55*img + 0.45*heatmap
 *      ImageNet normalization
 *  - ONNX output: logit (1,) => sigmoid(logit) => pASD
 */
public class InferencePipeline {

    public static float runPerImage(Bitmap stimulusBmp, List<Fixation> fixations, OnnxEnsembleRunner runner) throws Exception {
        // Build gaze_seq
        float[] gazeSeq = Preprocessing.buildGazeSeq(fixations);

        // Build stimulus_image tensor (CHW)
        float[] stimTensor = Preprocessing.buildStimulusImageTensor(stimulusBmp, fixations);

        // Run ONNX ensemble -> mean logit
        float meanLogit = runner.runEnsembleMeanLogit(gazeSeq, stimTensor);

        return ScoreAggregator.sigmoid(meanLogit);
    }
}
