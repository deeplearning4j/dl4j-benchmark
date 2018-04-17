package org.deeplearning4j.sets;

import org.deeplearning4j.benchmarks.BenchmarkCnn;
import org.deeplearning4j.memory.BenchmarkCnnMemory;
import org.deeplearning4j.memory.MemoryTest;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.nn.conf.WorkspaceMode;

public class StandardPerfBenchmarks {

    public static void main(String[] args) throws Exception {

        int testNum = 0;

        ModelType modelType;
        int[] batchSizes;
        int gcWindow = 10000;
        int totalIter = 20;

        switch (testNum){
            //MultiLayerNetwork tests:
            case 0:
                modelType = ModelType.ALEXNET;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;
            case 1:
                modelType = ModelType.ALEXNET;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;
            case 2:
                modelType = ModelType.VGG16;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;
            case 3:
                modelType = ModelType.VGG16;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;


            //ComputationGraph tests:
            case 4:
                modelType = ModelType.GOOGLELENET;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;
            case 5:
                modelType = ModelType.GOOGLELENET;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;
            case 6:
                modelType = ModelType.INCEPTIONRESNETV1;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;
            case 7:
                modelType = ModelType.INCEPTIONRESNETV1;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                break;

            default:
                throw new IllegalArgumentException("Invalid test: " + testNum);
        }

        for( int b : batchSizes) {

            BenchmarkCnn.main(new String[]{
                    "--modelType", modelType.toString(),
                    "--batchSize", String.valueOf(b),
                    "--gcWindow", String.valueOf(gcWindow),
                    "--totalIterations", String.valueOf(totalIter)
            });
        }

    }

}
