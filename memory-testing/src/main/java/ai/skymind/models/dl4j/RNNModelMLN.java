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
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RNNModelMLN implements BenchmarkModel {
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
                .layer(new SimpleRnn.Builder().nOut(64).activation(Activation.SIGMOID).build())
                .layer(new LSTM.Builder().nOut(64).activation(Activation.TANH).build())
                .layer(new Convolution1D.Builder().nOut(64).kernelSize(2).stride(1).convolutionMode(ConvolutionMode.Same).build())
                .layer(new GravesLSTM.Builder().nOut(64).activation(Activation.SOFTSIGN).build())
                .layer(new GravesBidirectionalLSTM.Builder().nOut(64).activation(Activation.TANH).build())
                .layer(new Bidirectional(new SimpleRnn.Builder().nOut(64).activation(Activation.RELU).build()))
                .layer(new RnnOutputLayer.Builder().nOut(10).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .setInputType(InputType.recurrent(64))
                .build();

        return new MultiLayerNetwork(conf);
    }
}
