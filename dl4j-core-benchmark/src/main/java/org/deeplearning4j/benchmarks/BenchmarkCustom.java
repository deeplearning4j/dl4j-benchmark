package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.ArrayUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.BenchmarkDataSetIterator;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Map;
import java.util.Random;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCustom extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.ALEXNET;
    //    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
//    public int numGPUs = 0;
    @Option(name="--datasetPath",usage="Path to the parent directly of multiple directories of classes of images.",aliases = "-dataset")
    public static String datasetPath = null;
    @Option(name="--numLabels",usage="Train batch size.",aliases = "-labels")
    public static int numLabels = -1;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-batch")
    public static int batchSize = 128;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 5000;
    @Option(name="--profile",usage="Run profiler and print results",aliases = "-profile")
    public static boolean profile = false;
    @Option(name="--cacheMode",usage="Cache mode setting for net")
    public static CacheMode cacheMode = CacheMode.NONE;
    @Option(name="--workspaceMode", usage="Workspace mode for net")
    public static WorkspaceMode workspaceMode = WorkspaceMode.SINGLE;
    @Option(name="--updater", usage="Updater for net")
    public static Updater updater = Updater.ADAM;

    private String datasetName  = "CUSTOM";
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

        if(datasetPath==null)
            throw new IllegalArgumentException("You must specify a valid path to a labelled dataset of images.");

        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType, null, numLabels, seed, iterations, workspaceMode, cacheMode, updater);

        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {
            int[][] inputShape = net.getValue().metaData().getInputShape();
            String description = datasetName + " " + batchSize + "x" + inputShape[0][0] + "x" + inputShape[0][1] + "x" + inputShape[0][2];
            log.info("Selected: " + net.getKey().toString() + " " + description);
            //            log.info("Preparing benchmarks for " + totalIterations + " iterations, " + numLabels + " labels");
            log.info("Preparing benchmarks for updater: {}, workspace: {}, cache mode: {}", updater, workspaceMode, cacheMode);

            log.info("Loading data...");
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            File mainPath = new File(datasetPath);
            FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));

            PathFilter pathFilter;
            if(numLabels>-1)
                pathFilter = new BalancedPathFilter(new Random(seed), labelMaker, 1000000, numLabels, 6000);
            else
                pathFilter = new RandomPathFilter(new Random(seed), NativeImageLoader.ALLOWED_FORMATS);

            InputSplit[] split = fileSplit.sample(pathFilter, 1.0);
            RecordReader trainRR = new ImageRecordReader(inputShape[0][2], inputShape[0][1], inputShape[0][0], labelMaker);
            trainRR.initialize(split[0]);
            DataSetIterator iter = new RecordReaderDataSetIterator(trainRR, batchSize);

            benchmark(net, description, numLabels, batchSize, seed, datasetName, iter, modelType, profile);
        }

        System.exit(0);
    }

    public static void main(String[] args) throws Exception {

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(false);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        new BenchmarkCustom().run(args);
    }
}
