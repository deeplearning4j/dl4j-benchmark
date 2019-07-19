package ai.skymind.pipeline;

import ai.skymind.Pipeline;
import ai.skymind.PipelineType;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class ImgRRPipeline implements Pipeline {

    @Override
    public PipelineType type() {
        return PipelineType.DATASET_ITERATOR;
    }

    @Override
    public DataSetIterator getIterator() {
        DataSetIterator iter = new TinyImageNetDataSetIterator(4, new int[]{64, 64}, DataSetType.TRAIN);
        //Normalize, and make random 10 class labels (tiny imagenet is 200 classes). We don't care about results
        // for these memory tests
        final Random r = new Random(12345);
        iter.setPreProcessor(new CompositeDataSetPreProcessor(
                        new ImagePreProcessingScaler(),
                        new DataSetPreProcessor() {
                            @Override
                            public void preProcess(DataSet dataSet) {
                                INDArray newLabels = Nd4j.create(DataType.FLOAT, 4, 10);
                                for( int i=0; i<4; i++ ){
                                    newLabels.putScalar(i, r.nextInt(10), 1.0);
                                }
                                dataSet.setLabels(newLabels);
                            }}));
        return iter;
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
