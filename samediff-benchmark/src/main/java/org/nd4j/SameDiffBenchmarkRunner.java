package org.nd4j;

import com.google.common.base.Preconditions;
import com.google.common.io.Files;
import lombok.extern.slf4j.Slf4j;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.models.SameDiffModel;
import org.nd4j.report.SDBenchmarkReport;
import org.nd4j.util.RemoteCachingLoader;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Slf4j
public class SameDiffBenchmarkRunner {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelClass", usage = "Model class for testing")
    public static String modelClass;
    @Option(name="--batchSize",usage="Train batch size.",aliases = "-batch")
    public static int batchSize = 32;
    @Option(name="--updater", usage="Updater for net")
    public static String updater = "adam";
    @Option(name="--numIterWarmup", usage="Updater for net")
    public static int numIterWarmup = 20;
    @Option(name="--numIter", usage="Updater for net")
    public static int numIter = 100;
    @Option(name="--l1", usage="L1 regularization coefficient for training")
    public static double l1 = 0.0;
    @Option(name="--l2", usage="L2 regularization coefficient for training")
    public static double l2 = 0.0;
    @Option(name="--wd", usage="Weight decay regularization coefficient for training")
    public static double wd = 0.0;

    public static void main(String... args) throws Exception {
        new SameDiffBenchmarkRunner().run(args);
    }

    public void run(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        try { parser.parseArgument(args); } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            System.exit(1);
        }

        Nd4j.getMemoryManager().togglePeriodicGc(false);
        RemoteCachingLoader.currentTestDir = Files.createTempDir();
        RemoteCachingLoader.currentTestDir.deleteOnExit();

        //Load model
        Preconditions.checkState(modelClass != null, "Model class was not specified");
        Class<?> c = Class.forName(modelClass);
        SameDiffModel model = (SameDiffModel) c.newInstance();
        SameDiff sd = model.getModel();

        //Configure model for training

        sd.setTrainingConfig(
                TrainingConfig.builder()
                        .updater(getUpdater(updater))
                        .regularization(regularization(l1, l2, wd))
                        .dataSetFeatureMapping(model.dataSetFeatureMapping())
                        .dataSetLabelMapping(model.dataSetLabelMapping())
                        .build()
        );

        //Create placeholder data
        Map<String,INDArray> phData = model.getPlaceholdersValues(batchSize);
        List<String> outputs = sd.outputs();

        SDBenchmarkReport r = new SDBenchmarkReport(modelClass, null);
        r.setBatchSize(batchSize);
        r.addTestConfig("# Iterations", numIter);
        r.addTestConfig("# Iterations (warmup)", numIterWarmup);
        r.addTestConfig("Updater", updater);
        r.addTestConfig("L1 Regularization", l1);
        r.addTestConfig("L2 Regularization", l2);
        r.addTestConfig("WD Regularization", wd);

        //Inference timing
        log.info("Starting inference timing...");
            //Warmup
        log.info("Warmup: {} iterations", numIterWarmup);
        for( int i=0; i<numIterWarmup; i++ ){
            Map<String,INDArray> out = sd.exec(phData, outputs);
        }

            //Testing
        System.gc();
        log.info("Testing: {} iterations", numIter);
        for( int i=0; i<numIter; i++ ){
            long start = System.currentTimeMillis();
            Map<String,INDArray> out = sd.exec(phData, outputs);
            long end = System.currentTimeMillis();
            r.addForwardTimeMs(end-start);
            System.gc();
        }

        //Some models can't be trained for one reason or another... for example batchnorm issues:
        if(model.trainable()) {
            //Backprop timing
            log.info("Starting backprop timing...");
            //Warmup
            log.info("Warmup: {} iterations", numIterWarmup);
            for (int i = 0; i < numIterWarmup; i++) {
                sd.execBackwards(phData);
            }

            //Testing
            System.gc();
            log.info("Testing: {} iterations", numIter);
            for (int i = 0; i < numIter; i++) {
                long start = System.currentTimeMillis();
                sd.execBackwards(phData);
                long end = System.currentTimeMillis();
                r.addGradientCalcTime(end - start);
                System.gc();
            }

            //Training timing
            MultiDataSet mds = createMds(model, phData);
            MultiDataSetIterator iter = new SingletonMultiDataSetIterator(mds);
            //Warmup
            log.info("Warmup: {} iterations", numIterWarmup);
            for (int i = 0; i < numIterWarmup; i++) {
                sd.fit(iter, 1);
            }

            //Testing
            System.gc();
            log.info("Testing: {} iterations", numIter);
            for (int i = 0; i < numIter; i++) {
                long start = System.currentTimeMillis();
                sd.fit(iter, 1);
                long end = System.currentTimeMillis();
                r.addForwardTimeMs(end - start);
                System.gc();
            }
        } else {
            log.warn("Skipping backprop/fit timing for benchmarks: model is not marked as trainable...");
        }
        log.info("Testing complete");

        String s = r.toString();
        System.out.println(s);
    }


    public static IUpdater getUpdater(String updaterName){
        //Note that for benchmarking purposes, the exact LR doesn't matter - same ops to be executed
        switch (updaterName.toLowerCase()){
            case "adadelta":
                return new AdaDelta();
            case "adam":
                return new Adam(1e-3);
            case "adamax":
                return new AdaMax(1e-3);
            case "amsgrad":
                return new AMSGrad(1e-3);
            case "nadam":
                return new Nadam(1e-3);
            case "nesterovs":
                return new Nesterovs(1e-3);
            case "rmsprop":
                return new RmsProp(1e-3);
            case "sgd":
                return new Sgd(1e-3);
        }
        throw new RuntimeException("Unknown or not implemented update: " + updaterName);
    }

    public static List<Regularization> regularization(double l1, double l2, double wd){
        if(l1 == 0.0 && l2 == 0.0 && wd == 0.0)
            return Collections.emptyList();

        Preconditions.checkState(l2 == 0.0 || wd == 0.0, "L2 and weight decay cannot both be used in the same model");
        List<Regularization> l = new ArrayList<>(2);
        if(l1 > 0.0)
            l.add(new L1Regularization(l1));
        if(l2 > 0.0)
            l.add(new L2Regularization(l2));
        if(wd > 0.0)
            l.add(new WeightDecay(wd, true));

        return l;
    }

    public MultiDataSet createMds(SameDiffModel model, Map<String,INDArray> phData){
        List<String> features = model.dataSetFeatureMapping();
        List<String> labels = model.dataSetLabelMapping();

        INDArray[] f = new INDArray[features.size()];
        INDArray[] l = new INDArray[labels.size()];
        for(int i=0; i<features.size(); i++ ){
            f[i] = phData.get(features.get(i));
        }

        for( int i=0; i<labels.size(); i++ ){
            l[i] = phData.get(labels.get(i));
        }

        return new MultiDataSet(f, l);
    }
}
