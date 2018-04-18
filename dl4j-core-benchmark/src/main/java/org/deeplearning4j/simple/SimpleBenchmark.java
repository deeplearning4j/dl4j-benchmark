package org.deeplearning4j.simple;

import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

public class SimpleBenchmark {

    public static void main(String[] args){


        Map<ModelType, TestableModel> networks = ModelSelector.select(ModelType.ALEXNET, null, 1000, 12345, 1, WorkspaceMode.SINGLE, CacheMode.NONE, Updater.ADAM);

        for( Map.Entry<ModelType, TestableModel> m : networks.entrySet()){

            MultiLayerNetwork net = (MultiLayerNetwork)m.getValue().init();
            int[] inputShape = new int[]{16,3,224,224};
            int[] labelShape = new int[]{16, 1000};
            INDArray input = Nd4j.create(inputShape);
            INDArray labels = Nd4j.create(labelShape);

            int nIter = 100;

            long start = System.currentTimeMillis();
            for( int i=0; i<nIter; i++ ){
                net.output(input);
            }
            long endOutput = System.currentTimeMillis();

            for( int i=0; i<nIter; i++ ){
                net.fit(input, labels);
            }
            long endFit = System.currentTimeMillis();

            double avgOutMs = (endOutput - start) / (double)nIter;
            double avgFitMs = (endFit - endOutput) / (double)nIter;
            System.out.println("Average output duration: " + avgOutMs);
            System.out.println("Average fit duration: " + avgFitMs);

        }

    }

}
