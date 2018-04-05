package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by kepricon on 17. 3. 30.
 * LeNet
 * Reference: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
 */

public class LeNet implements TestableModel {

    private int[] inputShape = new int[]{3,224,224};
    private int numLabels;
    private long seed;
    private int iterations;

    public LeNet(int numLabels, long seed, int iterations) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
    }

    public MultiLayerConfiguration conf() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .cacheMode(CacheMode.DEVICE)
                .seed(seed)
                .activation(Activation.IDENTITY)
//                .weightInit(WeightInit.XAVIER)
                .weightInit(WeightInit.DISTRIBUTION)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(5e-4)
                .updater(new NoOp())
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn1")
                        .nIn(inputShape[0])
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn2")
                        .nOut(50)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2}, new int[]{2, 2})
                        .name("maxpool2")
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX) // radial basis function required
                        .build())
                .setInputType(InputType.convolutionalFlat(inputShape[2],inputShape[1],inputShape[0]))
                .backprop(true).pretrain(false).build();

        return conf;
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

    public ModelMetaData metaData(){
        return new ModelMetaData(
                new int[][]{inputShape},
                1,
                ModelType.CNN
        );
    }
}