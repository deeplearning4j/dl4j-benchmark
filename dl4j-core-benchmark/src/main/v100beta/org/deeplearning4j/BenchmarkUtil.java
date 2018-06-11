package org.deeplearning4j;

import org.deeplearning4j.benchmarks.BenchmarkOp;
import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class BenchmarkUtil {

    /**
     * Workspace for working memory for a single layer: forward pass and backward pass
     * Note that this is opened/closed once per op (activate/backpropGradient call)
     */
    protected static final String WS_LAYER_WORKING_MEM = "WS_LAYER_WORKING_MEM";
    /**
     * Workspace for storing all layers' activations - used only to store activations (layer inputs) as part of backprop
     * Not used for inference
     */
    protected static final String WS_ALL_LAYERS_ACT = "WS_ALL_LAYERS_ACT";
    /**
     * Next 2 workspaces: used for:
     * (a) Inference: holds activations for one layer only
     * (b) Backprop: holds activation gradients for one layer only
     * In both cases, they are opened and closed on every second layer
     */
    protected static final String WS_LAYER_ACT_1 = "WS_LAYER_ACT_1";
    protected static final String WS_LAYER_ACT_2 = "WS_LAYER_ACT_2";
    /**
     * Workspace for working memory in RNNs - opened and closed once per RNN time step
     */
    protected static final String WS_RNN_LOOP_WORKING_MEM = "WS_RNN_LOOP_WORKING_MEM";


    protected static final WorkspaceConfiguration WS_LAYER_WORKING_MEM_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.3)
            .policyLearning(LearningPolicy.OVER_TIME)
            .policyReset(ResetPolicy.BLOCK_LEFT)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    protected static final WorkspaceConfiguration WS_ALL_LAYERS_ACT_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.2)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyReset(ResetPolicy.BLOCK_LEFT)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    protected static final WorkspaceConfiguration WS_LAYER_ACT_X_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.2)
            .policyLearning(LearningPolicy.OVER_TIME)
            .policyReset(ResetPolicy.BLOCK_LEFT)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    protected static final WorkspaceConfiguration WS_RNN_LOOP_WORKING_MEM_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0).overallocationLimit(0.2).policyReset(ResetPolicy.BLOCK_LEFT)
            .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
            .policyLearning(LearningPolicy.FIRST_LOOP).build();

    public static void enableRegularization(Model model){
        //No op for 1.0.0-alpha
    }

    public static long benchmark(BenchmarkOp op, INDArray input, INDArray labels, MultiLayerNetwork net) throws Exception {
        
        if(op == BenchmarkOp.FORWARD){
            return forwardTimeMultiLayerNetwork(input, net);
        } else if(op == BenchmarkOp.BACKWARD) {
            //Prepare network for backprop benchmark:
            //We need to do forward pass, and keep input activation arrays set on the layer input field, in the appropriate workspace


            LayerWorkspaceMgr mgr;
            if(net.getLayerWiseConfigurations().getTrainingWorkspaceMode() == WorkspaceMode.NONE){
                mgr = LayerWorkspaceMgr.noWorkspaces();
            } else {
                mgr = LayerWorkspaceMgr.builder()
                        .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                        .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                        .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .build();
            }

            try(MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS)){

                //protected List<INDArray> ffToLayerActivationsInWs(int layerIndex, @NonNull FwdPassType fwdPassType, boolean storeLastForTBPTT,
                //                                                      @NonNull INDArray input, INDArray fMask, INDArray lMask){

                Method m = MultiLayerNetwork.class.getDeclaredMethod("ffToLayerActivationsInWs", int.class, FwdPassType.class, boolean.class, INDArray.class, INDArray.class, INDArray.class);
                Object[] args = new Object[]{net.getnLayers()-1, FwdPassType.STANDARD, false, input, null, null};
                m.setAccessible(true);
                m.invoke(net, args);

                //calcBackpropGradients(null, true, false)
                //calcBackpropGradients(INDArray epsilon, boolean withOutputLayer, boolean tbptt) {
                //calcBackpropGradients(INDArray epsilon, boolean withOutputLayer, boolean tbptt, boolean returnInputActGrad)
                m = MultiLayerNetwork.class.getDeclaredMethod("calcBackpropGradients", INDArray.class, boolean.class, boolean.class, boolean.class);
                m.setAccessible(true);
                long start = System.nanoTime();
                m.invoke(net, null, true, false, false);
                return System.nanoTime() - start;
            }
        } else {
            long start = System.nanoTime();
            net.fit(input, labels);
            return System.nanoTime() - start;
        }
    }
    
    private static long forwardTimeMultiLayerNetwork(INDArray input, MultiLayerNetwork net){
        // forward
        long start = System.nanoTime();
        net.output(input);  //Note: output would probably be faster post ab_workspace_opt optimizations
        Nd4j.getExecutioner().commit();
        long time = System.nanoTime() - start;
        return time;
    }


    public static long benchmark(BenchmarkOp op, INDArray input, INDArray labels, ComputationGraph net) throws Exception {

        if(op == BenchmarkOp.FORWARD){
            long start = System.nanoTime();
            INDArray out = net.outputSingle(input);
            Nd4j.getExecutioner().commit();
            long time = System.nanoTime() - start;
            out.data().destroy();
            System.gc();
            return time;
        } else if(op == BenchmarkOp.BACKWARD){


            LayerWorkspaceMgr mgr;
            if(net.getConfiguration().getTrainingWorkspaceMode() == WorkspaceMode.NONE){
                mgr = LayerWorkspaceMgr.noWorkspaces();
            } else {
                mgr = LayerWorkspaceMgr.builder()
                        .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                        .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                        .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .build();
            }

            try (MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {

                //Prepare network for backprop benchmark:
                //Need to call method to generate activations in WS, ready for backprop
                Method m0 = ComputationGraph.class.getDeclaredMethod("getOutputLayerIndices");
                m0.setAccessible(true);
                int[] indices = (int[])m0.invoke(net);

                //ffToLayerActivationsInWS(boolean train, int layerIndex, int[] excludeIdxs, FwdPassType fwdPassType,
                //    boolean storeLastForTBPTT, INDArray[] input, INDArray[] fMask, INDArray[] lMask, boolean clearInputs) {
                Method m1 = ComputationGraph.class.getDeclaredMethod("ffToLayerActivationsInWS", boolean.class, int.class, int[].class,
                    FwdPassType.class, boolean.class, INDArray[].class, INDArray[].class, INDArray[].class, boolean.class);
                m1.setAccessible(true);
                m1.invoke(net, true, -1, indices, FwdPassType.STANDARD, true, new INDArray[]{input}, null, null, false);

                Method m = ComputationGraph.class.getDeclaredMethod("calcBackpropGradients", boolean.class, boolean.class, INDArray[].class);
                m.setAccessible(true);

                long start = System.nanoTime();
                m.invoke(net, true, false, null);
                long end = System.nanoTime();
                return end-start;
            }
        } else {
            long start = System.nanoTime();
            net.fit(new DataSet(input, labels));
            return System.nanoTime() - start;
        }

    }
}
