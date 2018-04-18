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
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public abstract class BaseMemoryBenchmark {

    private static final long MEM_RUNNABLE_ITER_FREQ_MS = 100;
    private static final AtomicLong maxMem = new AtomicLong(0);

    private static final int WARMUP_ITERS = 10;
    private static final int MEASURE_ITERS = 5;

    private static class MemoryRunnable implements Runnable {
        @Override
        public void run() {
            try{
                runHelper();
            } catch (Throwable t){
                log.error("Memory measuring runnable died", t);
                System.exit(1);
            }
        }

        private void runHelper() throws Exception {
            while(true){
                long curr = Pointer.physicalBytes();    //.totalBytes();
                if(curr > maxMem.get()){
                    maxMem.set(curr);
                }

                Thread.sleep(MEM_RUNNABLE_ITER_FREQ_MS);
            }
        }
    }

    public void benchmark(String name, String description, ModelType modelType, TestableModel testableModel, MemoryTest memoryTest,
                          List<Integer> batchSizes, WorkspaceMode workspaceMode) throws Exception {

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
        report.setMinibatchSizes(batchSizes);
        report.setWorkspaceMode(workspaceMode);

        if(mln != null){
            mln.getLayerWiseConfigurations().setTrainingWorkspaceMode(workspaceMode);
            mln.getLayerWiseConfigurations().setInferenceWorkspaceMode(workspaceMode);
        } else {
            cg.getConfiguration().setTrainingWorkspaceMode(workspaceMode);
            cg.getConfiguration().setInferenceWorkspaceMode(workspaceMode);
        }

        Thread.sleep(1000);
        long memAfter = maxMem.get();
        report.setBytesMaxPostInit(memAfter);

        int[] inputShape = testableModel.metaData().getInputShape()[0]; //TODO multi-input models



        if(memoryTest == MemoryTest.INFERENCE){
            boolean hitOOM = false;
            Map<Integer,Object> memUseVsMinibatch = new LinkedHashMap<>();
            report.setBytesForMinibatchInference(memUseVsMinibatch);

            for( int i=0; i<batchSizes.size(); i++ ){
                int[] inShape = new int[inputShape.length+1];
                inShape[0] = batchSizes.get(i);
                for(int j=0; j<inputShape.length; j++ ){
                    inShape[j+1] = inputShape[j];
                }

                log.info("Inference test: Starting minibatch size: {}", batchSizes.get(i));

                if(hitOOM){
                    memUseVsMinibatch.put(batchSizes.get(i), "OOM");
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

                        for( int iter=0; iter<MEASURE_ITERS; iter++){
                            INDArray input = Nd4j.create(inShape, 'c');
                            if(mln != null){
                                mln.output(input);
                            } else {
                                cg.outputSingle(input);
                            }

                            Thread.sleep(2 * MEM_RUNNABLE_ITER_FREQ_MS);
                        }

                        memUseVsMinibatch.put(batchSizes.get(i), maxMem.get());
                    } catch (Throwable e){
                        log.warn("Hit exception for minibatch size: {}", batchSizes.get(i), e);
                        hitOOM = true;
                        if(mln != null)
                            mln.clear();
                        if(cg != null)
                            cg.clear();
                        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                        System.gc();

                        memUseVsMinibatch.put(batchSizes.get(i), "OOM");
                    }
                }
            }
        } else if(memoryTest == MemoryTest.TRAINING){

            boolean hitOOM = false;
            Map<Integer,Object> memUseVsMinibatch = new LinkedHashMap<>();
            report.setBytesForMinibatchTrain(memUseVsMinibatch);

            int[] inShape = new int[inputShape.length+1];
            inShape[0] = 1;
            for(int j=0; j<inputShape.length; j++ ){
                inShape[j+1] = inputShape[j];
            }

            //Work out output size:
            int[] outShape;
            if(mln != null){
                outShape = mln.output(Nd4j.create(inShape)).shape();
            } else {
                outShape = cg.outputSingle(Nd4j.create(inShape)).shape();
            }


            for( int i=0; i<batchSizes.size(); i++ ){
                inShape[0] = batchSizes.get(i);
                outShape[0] = batchSizes.get(i);

                log.info("Training test: Starting minibatch size: {}", batchSizes.get(i));

                if(hitOOM){
                    memUseVsMinibatch.put(batchSizes.get(i), "OOM");
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
                            INDArray output = Nd4j.create(outShape, 'c');
                            if(mln != null){
                                mln.fit(input, output);
                            } else {
                                cg.fit(new org.nd4j.linalg.dataset.DataSet(input, output));
                            }
                            System.gc();
                        }

                        //Do measure iterations
                        maxMem.set(0);

                        for( int iter=0; iter<MEASURE_ITERS; iter++){
                            INDArray input = Nd4j.create(inShape, 'c');
                            INDArray output = Nd4j.create(outShape, 'c');
                            if(mln != null){
                                mln.fit(input, output);
                            } else {
                                cg.fit(new org.nd4j.linalg.dataset.DataSet(input, output));
                            }

                            Thread.sleep(2 * MEM_RUNNABLE_ITER_FREQ_MS);
                        }

                        memUseVsMinibatch.put(batchSizes.get(i), maxMem.get());
                    } catch (Throwable e){
                        log.warn("Hit exception for minibatch size: {}", batchSizes.get(i), e);
                        hitOOM = true;
                        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                        System.gc();

                        memUseVsMinibatch.put(batchSizes.get(i), "OOM");
                    }
                }
            }

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
