package org.deeplearning4j;

import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.workaround.TensorFlowCnnToFeedForwardPreProcessor;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.lang.reflect.Field;

public class VersionSpecificModels {


    public static ComputationGraph getPretrainedResnet50(WorkspaceMode wsm, CacheMode cm, Updater updater) {
        ComputationGraph cg;
        try{
           cg = (ComputationGraph)ResNet50.builder().build().initPretrained();
        } catch (IOException e){
            throw new RuntimeException(e);
        }

        //NOTE: Pretrained model is NOT trainable: it has a DenseLayer as the final layer, not an OutputLayer
        INDArray outW = cg.getLayer("fc1000").getParam("W");
        INDArray outB = cg.getLayer("fc1000").getParam("b");


        cg = new TransferLearning.GraphBuilder(cg)
                .removeVertexAndConnections("fc1000")
                .setWorkspaceMode(wsm)
                .addLayer("fc1000", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(2048).nOut(1000).activation(Activation.SOFTMAX).build(), "flatten_1")
                .setOutputs("fc1000")
                .build();

        cg.getConfiguration().setCacheMode(cm);
        cg.getLayer("fc1000").getParam("W").assign(outW);
        cg.getLayer("fc1000").getParam("b").assign(outB);

        if(BenchmarkReport.inferVersion().contains("1.0.0-beta")){
            //workaround for workspace issue on 1.0.0-beta
            PreprocessorVertex pv = (PreprocessorVertex) cg.getVertex("flatten_1");
            try {
                Field f = PreprocessorVertex.class.getDeclaredField("preProcessor");
                f.setAccessible(true);
                f.set(pv, new TensorFlowCnnToFeedForwardPreProcessor(1,1,2048));
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }

        return cg;
    }


}
