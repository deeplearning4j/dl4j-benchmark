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
import java.util.Collections;
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
                          DataSetIterator iter, ModelType modelType, boolean profile, int gcWindow, int occasionalGCFreq) throws Exception {


        log.info("=======================================");
        log.info("===== Benchmarking selected model =====");
        log.info("=======================================");

        //log.info("{}", VersionCheck.getVersionInfos());

        Model model = net.getValue().init();
        if(model == null){
            throw new IllegalStateException("Null model");
        }
        BenchmarkUtil.enableRegularization(model);

        BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), description);
        report.setModel(model);
        report.setBatchSize(batchSize);

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(gcWindow > 0);
        if(gcWindow > 0) {
            Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        }
        Nd4j.getMemoryManager().setOccasionalGcFrequency(occasionalGCFreq);

        report.setPeriodicGCEnabled(gcWindow > 0);
        report.setPeriodicGCFreq(gcWindow);
        report.setOccasionalGCFreq(occasionalGCFreq);

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

        model.setListeners(Collections.emptyList());

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

                //Backward
                long backwardTime = BenchmarkUtil.benchmark(BenchmarkOp.BACKWARD, input, labels, m);
                totalBackward += (backwardTime / 1e6);

                //Fit
                long fitTime = BenchmarkUtil.benchmark(BenchmarkOp.FIT, input, labels, m);
                totalFit += (fitTime / 1e6);

                nIterations++;
                if (nIterations % 100 == 0) log.info("Completed " + nIterations + " iterations");
            }
            profileEnd("Forward", profile);
        } else if (model instanceof ComputationGraph) {
            ComputationGraph g = (ComputationGraph)model;
            profileStart(profile);
            while (iter.hasNext()) {

                DataSet ds = iter.next();
                ds.migrate();
                INDArray input = ds.getFeatures();
                INDArray labels = ds.getLabels();

                // forward
                long forwardTime = BenchmarkUtil.benchmark(BenchmarkOp.FORWARD, input, labels, g);
                totalForward += (forwardTime / 1e6);

                //Backward
                long backwardTime = BenchmarkUtil.benchmark(BenchmarkOp.BACKWARD, input, labels, g);
                totalBackward += (backwardTime / 1e6);

                //Fit
                long fitTime = BenchmarkUtil.benchmark(BenchmarkOp.FIT, input, labels, g);
                totalFit += (fitTime / 1e6);

                nIterations++;
                if (nIterations % 100 == 0) log.info("Completed " + nIterations + " iterations");
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
