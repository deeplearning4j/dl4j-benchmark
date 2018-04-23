package org.deeplearning4j.sets;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.memory.BenchmarkCnnMemory;
import org.deeplearning4j.memory.MemoryTest;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

@Slf4j
public class StandardMemBenchmarks {

    @Option(name="--benchmark",usage="Benchmark number")
    public static int testNum = 0;


    public static void main(String[] args) throws Exception {
        new StandardMemBenchmarks().run(args);
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

        log.info("Starting test: {}", testNum);

        ModelType modelType;
        String batchSizes;
        MemoryTest memoryTest;
        WorkspaceMode workspaceMode;
        CacheMode cacheMode = CacheMode.NONE;
        Updater updater = Updater.ADAM;
        int gcWindow = 5000;

        switch (testNum){
            //MultiLayerNetwork tests:
            case 0:
                modelType = ModelType.ALEXNET;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = "1 2 4 8 16 32 64";
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 1:
                modelType = ModelType.ALEXNET;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = "1 2 4 8 16 32 64";
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 2:
                modelType = ModelType.VGG16;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = "1 2 4 8 16 32 64";
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 3:
                modelType = ModelType.VGG16;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = "1 2 4 8 16 32 64";
                workspaceMode = WorkspaceMode.SINGLE;
                break;


            //ComputationGraph tests:
            case 4:
                modelType = ModelType.GOOGLELENET;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = "1 2 4 8 16 32 64";
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 5:
                modelType = ModelType.GOOGLELENET;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = "1 2 4 8 16 32 64";
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 6:
                modelType = ModelType.INCEPTIONRESNETV1;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = "1 2 4 8 16 32 64";
//                batchSizes = "128 256 512";
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 7:
                modelType = ModelType.INCEPTIONRESNETV1;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = "1 2 4 8 16 32 64 128";
                workspaceMode = WorkspaceMode.SINGLE;
                break;

            default:
                throw new IllegalArgumentException("Invalid test: " + testNum);
        }

        BenchmarkCnnMemory.main(new String[]{
                "--modelType", modelType.toString(),
                "--batchSizes", batchSizes,
                "--memoryTest", memoryTest.toString(),
                "--workspaceMode", workspaceMode.toString(),
                "--cacheMode", cacheMode.toString(),
                "--updater", updater.toString(),
                "--gcWindow", String.valueOf(gcWindow)
        });

    }

}
