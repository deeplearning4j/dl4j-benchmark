package org.deeplearning4j.models.cnn;

import lombok.NoArgsConstructor;
import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A simple convolutional network for benchmarking purposes.
 *
 * @author Justin Long (crockpotveggies)
 */
@NoArgsConstructor
public class SimpleCNN implements TestableModel {

    private int[] inputShape = new int[] {3, 128, 128};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;
    private CacheMode cacheMode;

    public SimpleCNN(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode, CacheMode cacheMode) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = workspaceMode == WorkspaceMode.SINGLE ? ConvolutionLayer.AlgoMode.PREFER_FASTEST
                        : ConvolutionLayer.AlgoMode.NO_WORKSPACE;
        this.workspaceMode = workspaceMode;
        this.cacheMode = cacheMode;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                .seed(seed)
                                .activation(Activation.IDENTITY)
                                .weightInit(WeightInit.RELU)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .updater(new AdaDelta())
                                .convolutionMode(ConvolutionMode.Same)
                                .inferenceWorkspaceMode(workspaceMode)
                                .trainingWorkspaceMode(workspaceMode)
                                .cacheMode(cacheMode)
                                .list()
                                // block 1
                                .layer(0, new ConvolutionLayer.Builder(new int[] {1,1}).name("image_array")
                                                .nIn(inputShape[0]).nOut(16).build())
                                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX).build())
                                .layer(2, new OutputLayer.Builder().activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                                        .nOut(numLabels)
                                        .build())

                                .setInputType(InputType.convolutional(inputShape[2], inputShape[1],
                                                inputShape[0]))
                                .backprop(true).pretrain(false).build();

        return conf;
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ModelType.CNN);
    }
}
