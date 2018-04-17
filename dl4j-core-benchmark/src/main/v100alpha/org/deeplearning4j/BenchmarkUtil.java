package org.deeplearning4j;

import org.deeplearning4j.benchmarks.BenchmarkOp;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Method;

public class BenchmarkUtil {

    public static long benchmark(BenchmarkOp op, INDArray input, INDArray labels, MultiLayerNetwork net) throws Exception {
        if(op == BenchmarkOp.FORWARD){
            return forwardTimeMultiLayerNetwork(input, labels, net);
        } else if(op == BenchmarkOp.BACKWARD ) {
            //Prepare network for backprop benchmark:
            //We need to do forward pass, and
            // (a) keep input activation arrays set on the layer input field
            // (b) ensure input activation arrays are not defined in workspaces
            //To do this, we'll temporarily disable workspaces, then use the FF method that doesn't clear input arrays

            WorkspaceMode ws_train = net.getLayerWiseConfigurations().getTrainingWorkspaceMode();
            WorkspaceMode ws_inference = net.getLayerWiseConfigurations().getInferenceWorkspaceMode();
            net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.NONE);
            net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.NONE);
            net.setInput(input);
            net.setLabels(labels);
            //Ugly hack to support both 0.9.1 and 1.0.0-alpha and later...
            try {
                Method m = MultiLayerNetwork.class.getDeclaredMethod("feedForward", boolean.class, boolean.class);
//                        net.feedForward(true, false); //Train mode, don't clear inputs
                m.invoke(net, true, false);
            } catch (NoSuchMethodException e) {
                //Must be 0.9.1
                net.feedForward(true);
            } catch (Exception e){
                throw new RuntimeException(e);
            }

            net.getLayerWiseConfigurations().setTrainingWorkspaceMode(ws_train);
            net.getLayerWiseConfigurations().setInferenceWorkspaceMode(ws_inference);
            Nd4j.getExecutioner().commit();
            System.gc();


            // backward
            Method m = MultiLayerNetwork.class.getDeclaredMethod("backprop"); // requires reflection
            m.setAccessible(true);

            long start = System.nanoTime();
            m.invoke(net);
            Nd4j.getExecutioner().commit();
            long total = System.nanoTime() - start;
            return total;
        } else {
            long start = System.nanoTime();
            net.fit(input, labels);
            return System.nanoTime() - start;
        }
    }
    
    private static long forwardTimeMultiLayerNetwork(INDArray input, INDArray labels, MultiLayerNetwork net){
        // forward
        net.setInput(input);
        net.setLabels(labels);
        long start = System.nanoTime();
        net.feedForward();  //Note: output would probably be faster post ab_workspace_opt optimizations
        Nd4j.getExecutioner().commit();
        long time = System.nanoTime() - start;
        return time;
    }
}
