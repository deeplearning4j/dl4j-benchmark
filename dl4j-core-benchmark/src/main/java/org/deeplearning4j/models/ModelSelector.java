package org.deeplearning4j.models;

import com.beust.jcommander.ParameterException;
import org.deeplearning4j.models.cnn.*;
import org.deeplearning4j.models.cnn.VGG16;
import org.deeplearning4j.models.mlp.MLP;
import org.deeplearning4j.models.rnn.RNN;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.zoo.model.GoogLeNet;

import java.util.HashMap;
import java.util.Map;

/**
 * Helper class for easily selecting multiple models for benchmarking.
 */
public class ModelSelector {
    public static Map<ModelType,TestableModel> select(ModelType modelType, int[] inputShape, int numLabels, int seed, int iterations) {
        Map<ModelType,TestableModel> netmap = new HashMap<>();

        switch(modelType) {
            case ALL:
                netmap.putAll(ModelSelector.select(ModelType.CNN, null, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.RNN, null, numLabels, seed, iterations));
                break;
            // CNN models
            case CNN:
                netmap.putAll(ModelSelector.select(ModelType.ALEXNET, null, numLabels, seed, iterations));
                netmap.putAll(ModelSelector.select(ModelType.VGG16, null, numLabels, seed, iterations));
                break;
            case SIMPLECNN:
                netmap.put(ModelType.SIMPLECNN, new SimpleCNN(numLabels, seed, iterations));
                break;
            case ALEXNET:
                netmap.put(ModelType.ALEXNET, new AlexNet(numLabels, seed, iterations));
                break;
            case LENET:
                netmap.put(ModelType.LENET, new LeNet(numLabels, seed, iterations));
                break;
            case INCEPTIONRESNETV1:
                netmap.put(ModelType.INCEPTIONRESNETV1, new InceptionResNetV1(numLabels, seed, iterations));
                break;
            case FACENETNN4:
                netmap.put(ModelType.FACENETNN4, new FaceNetNN4(numLabels, seed, iterations));
                break;
            case VGG16:
                netmap.put(ModelType.VGG16, new VGG16(numLabels, seed, iterations));
                break;
            case MLP_SMALL:
                netmap.put(ModelType.MLP_SMALL, new MLP(inputShape[0], new int[]{512,512,512},numLabels, seed, Updater.ADAM ));
                break;

            case GOOGLELENET:
                netmap.put(ModelType.GOOGLELENET, new GoogleLeNet(numLabels, seed, iterations));
                break;

            // RNN models
            case RNN:
            case RNN_SMALL:
                netmap.put(ModelType.RNN_SMALL, new RNN(inputShape[0], new int[]{256,256},numLabels, seed, Updater.RMSPROP ));
                break;
            default:
//                // do nothing
        }

        if(netmap.size()==0) throw new ParameterException("Zero models have been selected for benchmarking.");

        return netmap;
    }
}
