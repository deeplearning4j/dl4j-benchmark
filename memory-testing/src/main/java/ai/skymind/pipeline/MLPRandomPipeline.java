package ai.skymind.pipeline;

import ai.skymind.Pipeline;
import ai.skymind.PipelineType;
import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Outputs [4, 64, 16] features, [4, 10, 16] labels
 */
public class MLPRandomPipeline implements Pipeline {
    @Override
    public PipelineType type() {
        return PipelineType.DATASET_ITERATOR;
    }

    @Override
    public DataSetIterator getIterator() {
        return new RandomDataSetIterator(100, new long[]{4, 64}, new long[]{4, 64}, RandomDataSetIterator.Values.RANDOM_UNIFORM,
                RandomDataSetIterator.Values.ONE_HOT);
    }

    @Override
    public MultiDataSetIterator getMdsIterator() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray[] getFeatures() {
        throw new UnsupportedOperationException("Not supported");
    }
}
