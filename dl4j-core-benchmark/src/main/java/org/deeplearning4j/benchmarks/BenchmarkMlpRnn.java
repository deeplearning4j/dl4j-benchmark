package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkMlpRnn extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. MLP_SMALL, RNN_SMALL).", aliases = "-model")
    public static ModelType modelType = ModelType.MLP_SMALL;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-batch")
    public static int trainBatchSize = 64;
    @Option(name="--timeSeriesLength",usage="Length of the time series data for RNNs",aliases = "-tsLength")
    public static int timeSeriesLength = 128;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 5000;
    @Option(name="--inputDimension",usage="The input size of the network",aliases = "-dim")
    public static int inputDimension = 256;
    @Option(name="--outputDimension",usage="The output size of the network",aliases = "-out")
    public static int outputDimension = 32;
    @Option(name="--numMinibatches",usage="The number of minibatches to use",aliases = "-minibatches")
    public static int numMinibatches = 64;
    @Option(name="--profile",usage="Run profiler and print results",aliases = "-profile")
    public static boolean profile = false;
    @Option(name="--cacheMode",usage="Cache mode setting for net")
    public static CacheMode cacheMode = CacheMode.NONE;
    @Option(name="--workspaceMode", usage="Workspace mode for net")
    public static WorkspaceMode workspaceMode = WorkspaceMode.SINGLE;
    @Option(name="--updater", usage="Updater for net")
    public static Updater updater = Updater.ADAM;

    private int seed = 42;

    public void run(String[] args) throws Exception {
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            System.exit(1);
        }

        // memory management optimizations
        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        switch(modelType){
            case MLP_SMALL:
            case RNN_SMALL:
                break;
            default:
                throw new UnsupportedOperationException("Image benchmarks are applicable to CNN models only.");
        }


        log.info("Preparing benchmarks for updater: {}, workspace: {}, cache mode: {}", updater, workspaceMode, cacheMode);
        log.info("Loading data...");
        //TODO export instead? This won't scale to large number of minibatches due to memory requirement
        List<DataSet> l = new ArrayList<>(numMinibatches);
        for( int i=0; i<numMinibatches; i++ ){
            INDArray f;
            INDArray labels;
            if(modelType == ModelType.MLP_SMALL){
                f = Nd4j.rand(trainBatchSize, inputDimension);
                labels = Nd4j.zeros(trainBatchSize, outputDimension);
                labels.getColumn(i%outputDimension).assign(1);
            } else {
                f = Nd4j.rand('f', new int[]{trainBatchSize, inputDimension, timeSeriesLength});
                labels = Nd4j.zeros(new int[]{trainBatchSize, outputDimension, timeSeriesLength}, 'f');
                labels.get(NDArrayIndex.all(), NDArrayIndex.point(i%outputDimension), NDArrayIndex.all()).assign(1);
            }
            l.add(new DataSet(f, labels));
        }

        DataSetIterator iter = new ExistingDataSetIterator(l);

        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType, new int[]{inputDimension}, outputDimension, seed, iterations, workspaceMode, cacheMode, updater);

        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {
            String description = net.getKey().toString()+" 1x"+inputDimension;
            benchmark(net, description, outputDimension, trainBatchSize, seed, "random", iter, modelType, profile);
        }
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkMlpRnn().run(args);
    }
}
