package org.deeplearning4j.models.cnn;


import org.deeplearning4j.VersionSpecificModels;
import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;


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