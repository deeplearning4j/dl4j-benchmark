package org.deeplearning4j.models.cnn;


import org.deeplearning4j.VersionSpecificModels;
import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class ResNet50Pretrained implements TestableModel {
    private int[] inputShape = new int[] { 3, 224, 224 };
    private WorkspaceMode workspaceMode;
    private CacheMode cacheMode;
    private Updater updater;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    public ResNet50Pretrained(WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
        this.cacheMode = cacheMode;
        this.workspaceMode = workspaceMode;
        this.updater = updater;
    }


    public ComputationGraph init(){
        return VersionSpecificModels.getPretrainedResnet50(workspaceMode, cacheMode, updater);
    }

    public ModelMetaData metaData(){
        return new ModelMetaData(
                new int[][]{inputShape},
                1,
                ModelType.CNN
        );
    }

}