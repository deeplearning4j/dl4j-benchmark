package ai.skymind;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public interface Pipeline {

    PipelineType type();

    DataSetIterator getIterator();

    MultiDataSetIterator getMdsIterator();

    INDArray[] getFeatures();

}
