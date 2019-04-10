package org.deeplearning4j.models.vae;

import lombok.AllArgsConstructor;
import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Alex on 13/04/2017.
 */
@AllArgsConstructor
public class VAESmall implements TestableModel {

    private int inputSize = 4;
    private int outputSize;
    private long seed;
    private Updater updater;
    private WorkspaceMode workspaceMode;
    private CacheMode cacheMode;

    public MultiLayerConfiguration conf(){
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(updater)
                .seed(seed)
                .activation(Activation.RELU)
                .inferenceWorkspaceMode(workspaceMode)
                .trainingWorkspaceMode(workspaceMode)
                .cacheMode(cacheMode)
                .list();

        builder.layer(0, new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(48)
                .decoderLayerSizes(48)
                .pzxActivationFunction(Activation.IDENTITY)
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))
                .nIn(inputSize)
                .nOut(outputSize)
                .build());

        return builder.build();
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

    public ModelMetaData metaData(){
        return new ModelMetaData(
                new int[][]{new int[]{inputSize}},
                1,
                ModelType.VAE_SMALL
        );
    }
}
