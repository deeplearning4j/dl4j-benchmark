package org.deeplearning4j.simple;

import org.deeplearning4j.benchmarks.BaseBenchmark;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

public class SimpleBenchmark {

    @Option(name = "--forward", usage = "Run forward pass in loop")
    public static boolean forward = false;

    @Option(name = "--fit", usage = "Run fit in loop")
    public static boolean fit = true;

    @Option(name = "--minibatch", usage = "minibatch size")
    public static int minibatch = 16;

    @Option(name = "--nIter", usage = "Number of iterations to run")
    public static int nIter=100;

    @Option(name="--updater", usage="Updater for net")
    public static Updater updater = Updater.ADAM;

    @Option(name="--model", usage="Model to test")
    public static ModelType modelType = ModelType.ALEXNET;

    @Option(name="--debugMode", usage="Enables ND4J debug mode")
    public static boolean debugMode = false;

    @Option(name="--profile", usage="Enables ND4J op profiler, and print results once done")
    public static boolean profile = false;

    public static void main(String[] args) throws Exception {
        new SimpleBenchmark().run(args);
    }

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

        System.out.println("Starting test: model=" + modelType + ", forward=" + forward + ", fit=" + fit + ", minibatch=" + minibatch +
                ", debugMode=" + debugMode + ", profile=" + profile);

        if(debugMode){
            Nd4j.getExecutioner().enableDebugMode(true);
        }

        Map<ModelType, TestableModel> networks = ModelSelector.select(modelType, null, 1000, 12345, 1, WorkspaceMode.SINGLE, CacheMode.NONE, updater);

        for (Map.Entry<ModelType, TestableModel> m : networks.entrySet()) {
            Model net = m.getValue().init();
            boolean isMln = net instanceof MultiLayerNetwork;
            MultiLayerNetwork mln = isMln ? (MultiLayerNetwork)net : null;
            ComputationGraph cg = isMln ? null : (ComputationGraph)net;

            int[] inputShapeNoBatch = m.getValue().metaData().getInputShape()[0];
            int[] inputShape = new int[inputShapeNoBatch.length+1];
            inputShape[0] = minibatch;
            for( int i=0; i<inputShapeNoBatch.length; i++ ){
                inputShape[i+1] = inputShapeNoBatch[i];
            }
            int[] labelShape = new int[]{minibatch, 1000};
            INDArray input = Nd4j.create(inputShape);
            INDArray labels = Nd4j.create(labelShape);

            long start = System.currentTimeMillis();
            if (forward) {
                BaseBenchmark.profileStart(profile);
                for (int i = 0; i < nIter; i++) {
                    if(isMln){
                        mln.output(input);
                    } else {
                        cg.outputSingle(input);
                    }
                }

                BaseBenchmark.profileEnd("Forward Pass", profile);
            }
            long endOutput = System.currentTimeMillis();

            if (fit) {
                BaseBenchmark.profileStart(profile);
                for (int i = 0; i < nIter; i++) {
                    if(isMln){
                        mln.fit(input, labels);
                    } else {
                        cg.fit(new DataSet(input, labels));
                    }
                }
                BaseBenchmark.profileEnd("Fit", profile);
            }
            long endFit = System.currentTimeMillis();

            double avgOutMs = (endOutput - start) / (double) nIter;
            double avgFitMs = (endFit - endOutput) / (double) nIter;
            if (forward) {
                System.out.println("Average output duration: " + avgOutMs);
            }
            if (fit) {
                System.out.println("Average fit duration: " + avgFitMs);
            }

            System.out.println("--- DONE ---");
        }

    }

}