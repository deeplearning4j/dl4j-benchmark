package org.nd4j.models;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.util.RemoteCachingLoader;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class InceptionResnetV2 implements SameDiffModel {
    @Override
    public SameDiff getModel() {
        try {
            File f = new ClassPathResource("tf_graphs/zoo_models/inception_resnet_v2_2018_04_27/tf_model.txt").getFile();
            return RemoteCachingLoader.LOADER.apply(f, "inception_resnet_v2");
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<String, INDArray> getPlaceholdersValues(int minibatch) {
        //Input shape is [mb, 128, 128, 3]
        Nd4j.getRandom().setSeed(12345);
        INDArray arr = Nd4j.rand(DataType.FLOAT, minibatch, 299, 299, 3);
        return Collections.singletonMap("input", arr);
    }

    @Override
    public List<String> dataSetFeatureMapping() {
        return Collections.singletonList("input");
    }

    @Override
    public List<String> dataSetLabelMapping() {
        return Collections.emptyList();
    }
}
