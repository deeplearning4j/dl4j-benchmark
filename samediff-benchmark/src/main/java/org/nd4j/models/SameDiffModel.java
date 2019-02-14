package org.nd4j.models;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

public interface SameDiffModel {

    SameDiff getModel();

    Map<String,INDArray> getPlaceholdersValues(int minibatch);

    List<String> dataSetFeatureMapping();

    List<String> dataSetLabelMapping();

    boolean trainable();

}
