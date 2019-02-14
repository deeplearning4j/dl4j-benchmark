package org.nd4j.models;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.util.RemoteCachingLoader;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class InceptionResnetV2 implements SameDiffModel {
    @Override
    public SameDiff getModel() {
        try {
            File f = new ClassPathResource("tf_graphs/zoo_models/inception_resnet_v2_2018_04_27/tf_model.txt").getFile();
            SameDiff sd = RemoteCachingLoader.LOADER.apply(f, "inception_resnet_v2_2018_04_27");

            //Convert all floating point constants to variables (was frozen by TF export)
            List<SDVariable> toConvert = new ArrayList<>();
            for(SDVariable v : sd.variables()){
                if(v.isConstant() && v.dataType().isFPType()){
                    toConvert.add(v);
                }
            }
            sd.convertToVariables(toConvert);
//            System.out.println(sd.summary());

            return sd;
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

    @Override
    public boolean trainable() {
        //NOT TRAINABLE due to batchnorm, until this is fixed: https://github.com/deeplearning4j/deeplearning4j/issues/7166
        return false;
    }
}
