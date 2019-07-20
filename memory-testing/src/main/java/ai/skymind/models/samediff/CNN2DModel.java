package ai.skymind.models.samediff;

import ai.skymind.SameDiffModel;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.LecunUniformInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 *
 * Memory guesstimate: based on roughly equivalent DL4J model, params + gradients + updater should be around 100kB
 * More will be kept in memory due to no workspaces...
 * Estimate: [4,8,64,64] x 40 = 21MB
 *
 * Suggested (tested, briefly) memory for CPU:
 * -Xmx256M -Dorg.bytedeco.javacpp.maxbytes=128M -Dorg.bytedeco.javacpp.maxphysicalbytes=576M
 * --dataClass ai.skymind.pipeline.ImgRRPipeline --modelClass ai.skymind.models.samediff.CNN2DModel
 */
public class CNN2DModel implements SameDiffModel {
    @Override
    public SameDiff getModel() {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, 4, 3, 64, 64);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 4, 10);

        SDVariable w1 = sd.var(Nd4j.rand(DataType.FLOAT, 2, 2, 3, 8));
        SDVariable l1 = sd.cnn().conv2d(in, w1, Conv2DConfig.builder().kH(2).kW(2).sH(1).sW(1).dataFormat("nchw").isSameMode(true).build());
        SDVariable l2 = sd.cnn().upsampling2d(l1, 2);
        SDVariable l3 = sd.cnn().avgPooling2d(l2, Pooling2DConfig.builder().kH(2).kH(2).sH(1).sW(1).isNHWC(true).isSameMode(true).build());
        SDVariable w4 = sd.var(Nd4j.rand(DataType.FLOAT, 2, 2, 2, 8));
        SDVariable l4 = sd.cnn().deconv2d(l3, w4, DeConv2DConfig.builder().kH(2).kW(2).dataFormat("nchw").isSameMode(true).build());
        SDVariable dw5 = sd.var("dw", new XavierInitScheme('c', 32, 32), DataType.FLOAT, 2, 2, 2, 2);
        SDVariable pw5 = sd.var("pw", new LecunUniformInitScheme('c', 32), DataType.FLOAT, 1, 1, 4, 8);
        SDVariable l5 = sd.cnn().separableConv2d(l4, dw5, pw5, null, Conv2DConfig.builder().kH(2).kW(2).sW(2).sH(2).isSameMode(true).dataFormat("nchw").build());
        SDVariable w6 = sd.var(Nd4j.rand(DataType.FLOAT, 2, 2, 8, 2));
        SDVariable l6 = sd.cnn().depthWiseConv2d(l5, w6, Conv2DConfig.builder().kH(2).kW(2).isSameMode(true).build());
        SDVariable l7 = sd.nn().relu(l6, 0);
//        SDVariable m8 = sd.var(Nd4j.rand(DataType.FLOAT, 16));
//        SDVariable v8 = sd.var(Nd4j.rand(DataType.FLOAT, 16));
//        SDVariable g8 = sd.var(Nd4j.rand(DataType.FLOAT, 16));
//        SDVariable b8 = sd.var(Nd4j.rand(DataType.FLOAT, 16));
//        SDVariable l8 = sd.nn().batchNorm("bn", l7, m8, v8, g8, b8, true, true, 1e-5, 1);
        SDVariable l9 = sd.nn().pad(l7, new int[][]{{0,0}, {0,0}, {1,1}, {1,1}}, 0.0);
        SDVariable l10 = sd.slice(l9, new int[]{0,0,1,1}, new int[]{4, 8, 64, 64});
        //TODO no locally connected...
        SDVariable l11 = sd.cnn().spaceToDepth(l10, 2, "nchw");
        SDVariable w12 = sd.var(Nd4j.rand(DataType.FLOAT, 2, 2, 32, 8));
        SDVariable l12 = sd.cnn().conv2d(l11, w12, Conv2DConfig.builder().kH(2).kW(2).sH(1).sW(1).dataFormat("nchw").isSameMode(true).build());

        SDVariable flatten = l12.reshape(4, -1);
        SDVariable sub = flatten.get(SDIndex.all(), SDIndex.interval(0, 256));
        SDVariable w13 = sd.var(Nd4j.rand(DataType.FLOAT, 256, 10));
        SDVariable b13 = sd.var(Nd4j.rand(DataType.FLOAT, 1, 10));
        SDVariable mmul = sub.mmul(w13).add(b13);
        SDVariable sofmax = sd.nn().softmax(mmul);
        SDVariable loss = sd.loss().logLoss(null, label, sofmax);


        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .updater(new Adam(0.001))
                .l2(0.001)
                .l1(0.001)
                .build());

        return sd;
    }
}
