package org.deeplearning4j.memory;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
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
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

import java.lang.reflect.Method;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseMemoryBenchmark {

    private static final long MEM_RUNNABLE_ITER_FREQ_MS = 100;
    private static final AtomicLong maxMem = new AtomicLong(0);
    private static final AtomicLong maxMemPhys = new AtomicLong(0);

    private static final int WARMUP_ITERS = 10;
    private static final int MEASURE_ITERS = 5;

    private static class MemoryRunnable implements Runnable {
        @Override
        public void run() {
            try{
                runHelper();
            } catch (Throwable t){
                t.printStackTrace();
            }
        }

        private void runHelper() throws Exception {
            while(true){
                long curr = Pointer.totalBytes();
                if(curr > maxMem.get()){
                    maxMem.set(curr);
                }

                curr = Pointer.physicalBytes();
                if(curr > maxMemPhys.get()){
                    maxMemPhys.set(curr);
                }

                Thread.sleep(MEM_RUNNABLE_ITER_FREQ_MS);
            }
        }
    }

    public void benchmark(String name, String description, ModelType modelType, TestableModel testableModel, MemoryTest memoryTest,
                          int[] minibatchSizes) throws Exception {

        new Thread(new MemoryRunnable()).start();

        log.info("=======================================");
        log.info("===== Benchmarking selected model =====");
        log.info("=======================================");

        MemoryBenchmarkReport report = new MemoryBenchmarkReport(name,description, memoryTest);


        Thread.sleep(1000);
        long memBefore = maxMem.get();
        report.setBytesMaxBeforeInit(memBefore);


        Model model = testableModel.init();
        MultiLayerNetwork mln = (model instanceof MultiLayerNetwork ? (MultiLayerNetwork)model : null);
        ComputationGraph cg = (model instanceof ComputationGraph ? (ComputationGraph)model : null);
        report.setModel(model);
        report.setMinibatchSizes(minibatchSizes);

        Thread.sleep(1000);
        long memAfter = maxMem.get();
        report.setBytesMaxPostInit(memAfter);

        int[] inputShape = testableModel.metaData().getInputShape()[0]; //TODO multi-input models



        if(memoryTest == MemoryTest.INFERENCE){
            boolean hitOOM = false;
            Map<Integer,Object> memUseVsMinibatch = new LinkedHashMap<>();
            for( int i=0; i<minibatchSizes.length; i++ ){
                int[] inShape = inputShape.clone();
                inShape[0] = minibatchSizes[i];


                if(hitOOM){
                    memUseVsMinibatch.put(minibatchSizes[i], "OOM");
                } else {
                    try{
                        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                        if(mln != null){
                            mln.clear();
                        }
                        if(cg != null){
                            cg.clear();
                        }
                        //Do warm-up iterations to initialize workspaces etc
                        for( int iter=0; iter<WARMUP_ITERS; iter++){
                            INDArray input = Nd4j.create(inShape, 'c');
                            if(mln != null){
                                mln.output(input);
                            } else {
                                cg.outputSingle(input);
                            }
                            System.gc();
                        }

                        //Do measure iterations
                        maxMem.set(0);
                        maxMemPhys.set(0);

                        for( int iter=0; iter<MEASURE_ITERS; iter++){
                            INDArray input = Nd4j.create(inShape, 'c');
                            if(mln != null){
                                mln.output(input);
                            } else {
                                cg.outputSingle(input);
                            }

                            Thread.sleep(2 * MEM_RUNNABLE_ITER_FREQ_MS);
                        }

                        memUseVsMinibatch.put(minibatchSizes[i], maxMem.get());
                    } catch (Exception e){
                        log.warn("Hit exception for minibatch size: {}", minibatchSizes[i], e);
                        hitOOM = true;
                        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                        System.gc();

                        memUseVsMinibatch.put(minibatchSizes[i], "OOM");
                    }
                }

                report.setBytesForMinibatchInference(memUseVsMinibatch);
            }
        } else if(memoryTest == MemoryTest.TRAINING){

            throw new UnsupportedOperationException("Not yet implemented");
        } else {
            throw new IllegalStateException("Unknown memory test: " + memoryTest);
        }



        log.info("=============================");
        log.info("===== Benchmark Results =====");
        log.info("=============================");

        System.out.println(report.getModelSummary());
        System.out.println(report.toString());
    }
}
