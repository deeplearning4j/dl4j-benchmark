package ai.skymind.models.samediff;

import ai.skymind.SameDiffModel;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;

public class MLPGenericModel implements SameDiffModel {

    @Override
    public SameDiff getModel() {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, 4, 64);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 4, 64);


        SDVariable w = sd.var(Nd4j.rand(DataType.FLOAT, 64, 64));
        SDVariable b = sd.var(Nd4j.rand(DataType.FLOAT, 1, 64));
        SDVariable z = in.mmul(w).add(b);
        SDVariable a = sd.nn().tanh(z);

        SDVariable l2 = sd.slice(a, new int[]{0, 32}, new int[]{4, 32});     //[4,32] out
        SDVariable l3 = sd.concat(1, l2, l2);                       //[4,64]
        SDVariable l4 = sd.nn().relu(l3.mmul(w).add(b), 0.0);

        SDVariable l5 = sd.nn().pad(l4, new int[][]{{0, 0}, {1, 2}}, 1.0);
        SDVariable l6 = l5.get(SDIndex.all(), SDIndex.interval(0, 64));
        SDVariable l7 = sd.nn().softmax(l6);
        SDVariable l7a = sd.zerosLike(l7);
        SDVariable l7b = sd.onesLike(l7);
        SDVariable l8 = l7.add(l7a).mul(l7b);
        SDVariable l9 = l8.sub(sd.rank(l8).mul(0).reshape(1, 1));
        SDVariable l10 = l9.castTo(DataType.DOUBLE).castTo(DataType.HALF).castTo(DataType.FLOAT);
        SDVariable l11 = sd.math().reciprocal(l10).permute(1,0).permute(1,0);
        SDVariable l12 = sd.math().log(l11);
        SDVariable l13 = sd.nn().sigmoid(sd.math().mergeAdd(l12, l11));
        SDVariable l14 = sd.nn().swish(l13);
        SDVariable l15 = sd.random().bernoulli("bernoulli", 0.5, l14.shape()).castTo(DataType.FLOAT).add(l14);
        SDVariable l16 = sd.random().normalTruncated(1, 0.5, 4, 64).castTo(DataType.FLOAT).add(l15);

        SDVariable l2Loss = sd.loss().l2Loss(l16);
        SDVariable loss = sd.loss().meanSquaredError("loss", label, l16);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .updater(new Nesterovs(0.001))
                .l2(0.001)
                .l1(0.001)
                .build());

        return sd;
    }

}
