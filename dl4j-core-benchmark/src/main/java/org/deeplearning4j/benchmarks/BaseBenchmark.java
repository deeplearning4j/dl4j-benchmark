package org.deeplearning4j.benchmarks;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BenchmarkUtil;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.listeners.BenchmarkListener;
import org.deeplearning4j.listeners.BenchmarkReport;
import org.deeplearning4j.listeners.MemoryReportingListener;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

import java.util.*;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseBenchmark {
    protected int listenerFreq = 10;
    protected int iterations = 1;
    protected static Map<ModelType, TestableModel> networks;
    protected boolean train = true;

    @Builder(builderClassName = "Benchmark", buildMethodName = "execute")
    public void benchmark(Map.Entry<ModelType, TestableModel> net, String description, int numLabels, int batchSize, int seed, String datasetName,
                          DataSetIterator iter, ModelType modelType, boolean profile, int gcWindow, int occasionalGCFreq,
                          boolean usePW, int pwNumThreads, int pwAvgFreq, int pwPrefetchBuffer, boolean memoryListener) throws Exception {


        log.info("=======================================");
        log.info("===== Benchmarking selected model =====");
        log.info("=======================================");

        //log.info("{}", VersionCheck.getVersionInfos());

        Model model = net.getValue().init();
        if(model == null){
            throw new IllegalStateException("Null model");
        }
        BenchmarkUtil.enableRegularization(model);

        if(usePW && pwNumThreads < 0){
            Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
            List<HashMap<Object,Object>> cudaDevices = (List<HashMap<Object, Object>>) p.get("cuda.devicesInformation");
            if (cudaDevices == null) {
                throw new IllegalStateException("Cannot infer number of GPUs for setting pwNumThreads config." +
                        "Set this config value manually (currently: " + pwNumThreads + ")");
            }
            pwNumThreads = cudaDevices.size();
            log.info("ParallelWrapper: Set to {} devices/threads", pwNumThreads);
        }

        BenchmarkReport report = new BenchmarkReport(net.getKey().toString(), description);
        report.setModel(model);
        report.setBatchSize(batchSize);
        report.setParallelWrapper(usePW);
        report.setParallelWrapperNumThreads(pwNumThreads);

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(gcWindow > 0);
        if(gcWindow > 0) {
            Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        }
        Nd4j.getMemoryManager().setOccasionalGcFrequency(occasionalGCFreq);

        report.setPeriodicGCEnabled(gcWindow > 0);
        report.setPeriodicGCFreq(gcWindow);
        report.setOccasionalGCFreq(occasionalGCFreq);

        ParallelWrapper pw = null;
        if(usePW){
            pw = new ParallelWrapper.Builder(model)
                    .prefetchBuffer(pwPrefetchBuffer)
                    .workers(pwNumThreads)
                    .averagingFrequency(pwAvgFreq)
                    .reportScoreAfterAveraging(true)
                    .build();
        }

        //Warm-up
        log.info("===== Warming up =====");
        if(!usePW) {
            DataSetIterator warmup = new EarlyTerminationDataSetIterator(iter, 10);
            if (model instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) model).fit(warmup);
            } else if (model instanceof ComputationGraph) {
                ((ComputationGraph) model).fit(warmup);
            }
        } else {
            DataSetIterator warmup = new EarlyTerminationDataSetIterator(iter, 10 * pwNumThreads);
            pw.fit(warmup);
        }
        iter.reset();

        val listeners = Arrays.asList(
                (memoryListener ? new MemoryReportingListener() : new PerformanceListener(1)), new BenchmarkListener(report));
        if(!usePW){
            model.setListeners(listeners);
        } else {
            pw.setListeners(listeners);
        }

        log.info("===== Benchmarking training iteration =====");
        profileStart(profile);
        if(!usePW) {
            if (model instanceof MultiLayerNetwork) {
                // timing
                ((MultiLayerNetwork) model).fit(iter);
            } else if (model instanceof ComputationGraph) {
                // timing
                ((ComputationGraph) model).fit(iter);
            }
        } else {
            log.info("--- Benchmarking using ParallelWrapper, {} threads ---", pwNumThreads);
            pw.fit(iter);
        }
        profileEnd("Fit", profile);


        if(!usePW) {
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
                MultiLayerNetwork m = (MultiLayerNetwork) model;
                profileStart(profile);
                while (iter.hasNext()) {
                    DataSet ds = iter.next();
                    INDArray input = ds.getFeatures();
                    INDArray labels = ds.getLabels();

                    // forward
                    long forwardTime = BenchmarkUtil.benchmark(BenchmarkOp.FORWARD, input, labels, m);
                    totalForward += (forwardTime / 1e6);
                    System.gc();

                    //Backward
                    long backwardTime = BenchmarkUtil.benchmark(BenchmarkOp.BACKWARD, input, labels, m);
                    totalBackward += (backwardTime / 1e6);
                    System.gc();

                    //Fit
                    long fitTime = BenchmarkUtil.benchmark(BenchmarkOp.FIT, input, labels, m);
                    totalFit += (fitTime / 1e6);
                    System.gc();

                    nIterations++;
                    if (nIterations % 100 == 0) log.info("Completed " + nIterations + " iterations");
                }
                profileEnd("Forward", profile);
            } else if (model instanceof ComputationGraph) {
                ComputationGraph g = (ComputationGraph) model;
                profileStart(profile);
                while (iter.hasNext()) {

                    DataSet ds = iter.next();
                    ds.migrate();
                    INDArray input = ds.getFeatures();
                    INDArray labels = ds.getLabels();

                    // forward
                    long forwardTime = BenchmarkUtil.benchmark(BenchmarkOp.FORWARD, input, labels, g);
                    totalForward += (forwardTime / 1e6);
                    System.gc();

                    //Backward
                    long backwardTime = BenchmarkUtil.benchmark(BenchmarkOp.BACKWARD, input, labels, g);
                    totalBackward += (backwardTime / 1e6);
                    System.gc();

                    //Fit
                    long fitTime = BenchmarkUtil.benchmark(BenchmarkOp.FIT, input, labels, g);
                    totalFit += (fitTime / 1e6);
                    System.gc();

                    nIterations++;
                    if (nIterations % 100 == 0) log.info("Completed " + nIterations + " iterations");
                }
                profileEnd("Backward", profile);
            }
            report.setAvgFeedForward(totalForward / (double) nIterations);
            report.setAvgBackprop(totalBackward / (double) nIterations);
            report.setAvgFit(totalFit / (double) nIterations);
        }


        log.info("=============================");
        log.info("===== Benchmark Results =====");
        log.info("=============================");

        System.out.println(report.getModelSummary());
        System.out.println(report.toString());
    }

    public static void profileStart(boolean enabled) {
        if (enabled) {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ALL);
            OpProfiler.getInstance().reset();
        }
    }

    public static void profileEnd(String label, boolean enabled) {
        if (enabled) {
            log.info("==== " + label + " - OpProfiler Results ====");
            OpProfiler.getInstance().printOutDashboard();
            OpProfiler.getInstance().reset();
        }
    }
}
