package ai.skymind.models.dl4j;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class CNN2DModelCG extends CNN2DModelMLN {

    @Override
    public Model getModel() {
        return ((MultiLayerNetwork)super.getModel()).toComputationGraph();
    }

}
