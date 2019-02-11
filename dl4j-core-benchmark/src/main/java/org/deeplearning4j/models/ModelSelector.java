package org.deeplearning4j.models;

import com.beust.jcommander.ParameterException;
import org.deeplearning4j.models.cnn.*;
import org.deeplearning4j.models.mlp.MLP;
import org.deeplearning4j.models.rnn.RNN;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;

import java.util.HashMap;
import java.util.Map;

/**
 * Helper class for easily selecting multiple models for benchmarking.
 */
public class ModelSelector {
    public static Map<ModelType,TestableModel> select(ModelType modelType, int[] inputShape, int numLabels, int seed,
                                                      int iterations, WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
        Map<ModelType,TestableModel> netmap = new HashMap<>();

        switch(modelType) {
            case ALL:
                netmap.putAll(ModelSelector.select(ModelType.CNN, null, numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                netmap.putAll(ModelSelector.select(ModelType.RNN, null, numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            // CNN models
            case CNN:
                netmap.putAll(ModelSelector.select(ModelType.ALEXNET, null, numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                netmap.putAll(ModelSelector.select(ModelType.VGG16, null, numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case SIMPLECNN:
                netmap.put(ModelType.SIMPLECNN, new SimpleCNN(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case ALEXNET:
                netmap.put(ModelType.ALEXNET, new AlexNet(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case LENET:
                netmap.put(ModelType.LENET, new LeNet(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case INCEPTIONRESNETV1:
                netmap.put(ModelType.INCEPTIONRESNETV1, new InceptionResNetV1(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case FACENETNN4:
                netmap.put(ModelType.FACENETNN4, new FaceNetNN4(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case VGG16:
                netmap.put(ModelType.VGG16, new VGG16(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case MLP_SMALL:
                netmap.put(ModelType.MLP_SMALL, new MLP(inputShape[0], new int[]{512,512,512},numLabels, seed, updater, workspaceMode, cacheMode));
                break;
            case RESNET50:
                netmap.put(ModelType.RESNET50, new ResNet50(numLabels, seed, iterations, workspaceMode, cacheMode, updater));
                break;
            case RESNET50PRE:
                netmap.put(ModelType.RESNET50PRE, new ResNet50Pretrained(workspaceMode, cacheMode, updater));
                break;
            // RNN models
            case RNN:
            case RNN_SMALL:
                netmap.put(ModelType.RNN_SMALL, new RNN(inputShape[0], new int[]{256,256},numLabels, seed, updater, workspaceMode, cacheMode ));
                break;
            default:
//                // do nothing
        }

        if(netmap.size()==0) throw new ParameterException("Zero models have been selected for benchmarking.");

        return netmap;
    }
}
