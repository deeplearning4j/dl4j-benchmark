package ai.skymind.models.dl4j;

import ai.skymind.BenchmarkModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

public class CNN2DModelMLN implements BenchmarkModel {
    @Override
    public Model getModel() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .l1(0.001)
                .dropOut(0.5)
                .weightNoise(new WeightNoise(new NormalDistribution(0.0, 0.01), true))
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(new ConvolutionLayer.Builder().nOut(8).kernelSize(2,2).stride(1,1).padding(0,0).dilation(1,1).build())
                .layer(new Upsampling2D.Builder().size(2).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(1,1).build())
                .layer(new Deconvolution2D.Builder().nOut(8).kernelSize(2,2).stride(1,1).build())
                .layer(new SeparableConvolution2D.Builder().nOut(8).kernelSize(2,2).stride(2,2).depthMultiplier(2).build())
                .layer(new DepthwiseConvolution2D.Builder().nOut(8).kernelSize(2,2).stride(1,1).depthMultiplier(2).build())
                .layer(new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new ZeroPaddingLayer(1,1))
                .layer(new Cropping2D(1,1))
                .layer(new LocallyConnected2D.Builder().nOut(8).kernelSize(2,2).stride(1,1).build())
                .layer(new SpaceToDepthLayer.Builder().blocks(2).build())
                .layer(new ConvolutionLayer.Builder().activation(Activation.TANH).kernelSize(2,2).stride(2,2).nOut(2).build())
                .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX).nOut(10).build())
                .setInputType(InputType.convolutional(64, 64, 3))
                .build();

        return new MultiLayerNetwork(conf);
    }
}
