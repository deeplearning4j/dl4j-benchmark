package ai.skymind.pipeline;

import ai.skymind.Pipeline;
import ai.skymind.PipelineType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MLPEmbeddingArraysPipeline implements Pipeline {
    @Override
    public PipelineType type() {
        return PipelineType.INDARRAYS;
    }

    @Override
    public DataSetIterator getIterator() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public MultiDataSetIterator getMdsIterator() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray[] getFeatures() {
        INDArray arr = Nd4j.rand(DataType.FLOAT, 4, 1).muli(64);
        Transforms.round(arr, false);
        Transforms.min(arr, 63, false);
        return new INDArray[]{arr};
    }
}
