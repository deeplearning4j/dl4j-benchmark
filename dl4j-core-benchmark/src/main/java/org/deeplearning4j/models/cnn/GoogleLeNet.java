package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.zoo.model.GoogLeNet;

public class GoogleLeNet implements TestableModel {

    private int[] inputShape = new int[]{3,224,224};
    private int numLabels;
    private long seed;
    private int iterations;
    private CacheMode cacheMode;
    private Updater updater;

    public GoogleLeNet(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.cacheMode = cacheMode;
        this.updater = updater;
    }

    @Override
    public Model init() {
//        return new GoogLeNet(numLabels, seed, iterations).init();
//        return new GoogLeNet(numLabels, seed, WorkspaceMode.SINGLE).init();
        return null;    //TODO
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(
                new int[][]{inputShape},
                1,
                ModelType.CNN
        );
    }
}
