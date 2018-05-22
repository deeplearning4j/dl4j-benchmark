package org.deeplearning4j.simple;

import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

public class SimpleBenchmark {

    @Option(name = "--forward", usage = "Run forward pass in loop", aliases = "-fwd")
    public static boolean forward = false;

    @Option(name = "--fit", usage = "Run fit in loop", aliases = "-fwd")
    public static boolean fit = true;

    @Option(name = "--minibatch", usage = "minibatch size")
    public static int minibatch = 16;

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

        System.out.println("Starting test: forward=" + forward + ", fit=" + fit + ", minibatch=" + minibatch);

        Map<ModelType, TestableModel> networks = ModelSelector.select(ModelType.ALEXNET, null, 1000, 12345, 1, WorkspaceMode.SINGLE, CacheMode.NONE, Updater.ADAM);

        for (Map.Entry<ModelType, TestableModel> m : networks.entrySet()) {

            MultiLayerNetwork net = (MultiLayerNetwork) m.getValue().init();
            int[] inputShape = new int[]{minibatch, 3, 224, 224};
            int[] labelShape = new int[]{minibatch, 1000};
            INDArray input = Nd4j.create(inputShape);
            INDArray labels = Nd4j.create(labelShape);

            int nIter = 100;

            long start = System.currentTimeMillis();
            if (forward) {
                for (int i = 0; i < nIter; i++) {
                    net.output(input);
                }
            }
            long endOutput = System.currentTimeMillis();

            if (fit) {
                for (int i = 0; i < nIter; i++) {
                    net.fit(input, labels);
                }
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