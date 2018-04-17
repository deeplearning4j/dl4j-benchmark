package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BenchmarkUtil;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.listeners.BenchmarkListener;
import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.listeners.TrainingDiscriminationListener;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

import java.lang.reflect.Method;
import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseBenchmark {
    protected int listenerFreq = 10;
    protected int iterations = 1;
    protected static Map<ModelType, TestableModel> networks;
    protected boolean train = true;

    public void benchmark(Map.Entry<ModelType, TestableModel> net, String description, int numLabels, int batchSize, int seed, String datasetName,
                          DataSetIterator iter, ModelType modelType, boolean profile) throws Exception {

        log.info("=======================================");
        log.info("===== Benchmarking selected model =====");
        log.info("=======================================");

        //log.info("{}", VersionCheck.getVersionInfos());

        Model model = net.getValue().init();
        BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), description);
        report.setModel(model);
        report.setBatchSize(batchSize);

        // ADSI
        AsyncDataSetIterator asyncIter = new AsyncDataSetIterator(iter, 2, true);

        for (int i = 0; i < 1; i++) {
            if (asyncIter.hasNext()) {
                DataSet ds = asyncIter.next();
                if (model instanceof MultiLayerNetwork) {
                    ((MultiLayerNetwork) model).fit(ds);
                } else if (model instanceof ComputationGraph) {
                    ((ComputationGraph) model).fit(ds);
                }
            }
        }

        model.setListeners(new PerformanceListener(1), new BenchmarkListener(report), new TrainingDiscriminationListener());

        log.info("===== Benchmarking training iteration =====");
        profileStart(profile);
        if (model instanceof MultiLayerNetwork) {
            // timing
            ((MultiLayerNetwork) model).fit(asyncIter);
        }
        if (model instanceof ComputationGraph) {
            // timing
            ((ComputationGraph) model).fit(asyncIter);
        }
        profileEnd("Fit", profile);


        log.info("===== Benchmarking forward/backward pass =====");
        /*
            Notes: popular benchmarks will measure the time it takes to set the input and feed forward
            and backward. This is consistent with benchmarks seen in the wild like this code:
            https://github.com/jcjohnson/cnn-benchmarks/blob/master/cnn_benchmark.lua
         */
        iter.reset();

        long totalForward = 0;
        long totalBackward = 0;
        long totalFit = 0;
        long nIterations = 0;
        if (model instanceof MultiLayerNetwork) {
            MultiLayerNetwork m = (MultiLayerNetwork)model;
            profileStart(profile);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                ds.migrate();
                INDArray input = ds.getFeatures();
                INDArray labels = ds.getLabels();

                // forward
                long forwardTime = BenchmarkUtil.benchmark(BenchmarkOp.FORWARD, input, labels, m);
                totalForward += (forwardTime / 1e6);

                long backwardTime = BenchmarkUtil.benchmark(BenchmarkOp.BACKWARD, input, labels, m);
                totalBackward += (backwardTime / 1e6);

                long fitTime = BenchmarkUtil.benchmark(BenchmarkOp.FIT, input, labels, m);
                totalFit += (fitTime / 1e6);

                nIterations++;
                if (nIterations % 100 == 0) log.info("Completed " + nIterations + " iterations");
            }
            profileEnd("Forward", profile);
        } else if (model instanceof ComputationGraph) {
            profileStart(profile);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                INDArray input = ds.getFeatures();
                INDArray labels = ds.getLabels();

//                try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace("LOOP_EXTERNAL")) {

                // forward
                ((ComputationGraph) model).setInput(0, input);
                ((ComputationGraph) model).setLabels(labels);
                long forwardTime = System.nanoTime();
                ((ComputationGraph) model).feedForward();
                Nd4j.getExecutioner().commit();
                forwardTime = System.nanoTime() - forwardTime;
                totalForward += (forwardTime / 1e6);

                //Prepare network for backprop benchmark:
                //We need to do forward pass, and
                // (a) keep input activation arrays set on the layer input field
                // (b) ensure input activation arrays are not defined in workspaces
                //To do this, we'll temporarily disable workspaces, then use the FF method that doesn't clear input arrays

                WorkspaceMode ws_train = ((ComputationGraph) model).getConfiguration().getTrainingWorkspaceMode();
                WorkspaceMode ws_inference = ((ComputationGraph) model).getConfiguration().getInferenceWorkspaceMode();
                ((ComputationGraph) model).getConfiguration().setTrainingWorkspaceMode(WorkspaceMode.NONE);
                ((ComputationGraph) model).getConfiguration().setInferenceWorkspaceMode(WorkspaceMode.NONE);
                ((ComputationGraph) model).setInput(0, input);
                ((ComputationGraph) model).setLabels(labels);
                try {
                    Method m = ComputationGraph.class.getDeclaredMethod("feedForward", INDArray[].class, boolean.class, boolean.class);
                    //((ComputationGraph) model).feedForward(new INDArray[]{input}, true, false); //Train mode, don't clear inputs
                    m.invoke(model, new INDArray[]{input}, true, false);
                } catch (NoSuchMethodException e) {
                    //Must be 0.9.1
                    ((ComputationGraph) model).feedForward(new INDArray[]{input}, true);
                }
                ((ComputationGraph) model).getConfiguration().setTrainingWorkspaceMode(ws_train);
                ((ComputationGraph) model).getConfiguration().setInferenceWorkspaceMode(ws_inference);
                Nd4j.getExecutioner().commit();
                System.gc();

                // backward
                long backwardTime = System.nanoTime();
                Method m = ComputationGraph.class.getDeclaredMethod("calcBackpropGradients", boolean.class, INDArray[].class);
                m.setAccessible(true);
                m.invoke(model, false, null);
                Nd4j.getExecutioner().commit();
                backwardTime = System.nanoTime() - backwardTime;
                totalBackward += (backwardTime / 1e6);

                nIterations += 1;
                if (nIterations % 100 == 0) log.info("Completed " + nIterations + " iterations");
//                }
            }
            profileEnd("Backward", profile);
        }
        report.setAvgFeedForward(totalForward / (double) nIterations);
        report.setAvgBackprop(totalBackward / (double) nIterations);
        report.setAvgFit(totalFit / (double) nIterations);


        log.info("=============================");
        log.info("===== Benchmark Results =====");
        log.info("=============================");

        System.out.println(report.getModelSummary());
        System.out.println(report.toString());
    }

    private static void profileStart(boolean enabled) {
        if (enabled) {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);
            OpProfiler.getInstance().reset();
        }
    }

    private static void profileEnd(String label, boolean enabled) {
        if (enabled) {
            log.info("==== " + label + " - OpProfiler Results ====");
            OpProfiler.getInstance().printOutDashboard();
        }
    }
}
