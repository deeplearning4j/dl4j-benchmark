package ai.skymind.models.dl4j;

import ai.skymind.BenchmarkModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.constraint.MaxNormConstraint;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;

public class MLPModelMLN implements BenchmarkModel {
    @Override
    public Model getModel() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(0.01))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .l1(0.001)
                .dropOut(new GaussianDropout(0.5))
                .constrainAllParameters(new MaxNormConstraint(1.0))
                .weightNoise(new DropConnect(0.5))
                .list()
                .layer(new EmbeddingLayer.Builder().nIn(64).nOut(64).activation(Activation.SWISH).build())
                .layer(new DenseLayer.Builder().nIn(64).nOut(64).activation(Activation.HARDSIGMOID).build())
                .layer(new VariationalAutoencoder.Builder().nOut(64).encoderLayerSizes(128,64).activation(Activation.TANH).pzxActivationFunction(Activation.SOFTSIGN).build())
                .layer(new BatchNormalization.Builder().build())
                .layer(new AutoEncoder.Builder().nOut(64).activation(Activation.GELU).build())
                .layer(new DenseLayer.Builder().nOut(10).activation(Activation.IDENTITY).build())
                .layer(new LossLayer.Builder().activation(Activation.SOFTMAX).build())
                .setInputType(InputType.feedForward(1))
                .build();

        return new MultiLayerNetwork(conf);


    }
}
