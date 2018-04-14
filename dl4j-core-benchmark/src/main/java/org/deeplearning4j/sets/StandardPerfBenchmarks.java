package org.deeplearning4j.sets;

import org.deeplearning4j.benchmarks.BenchmarkCnn;
import org.deeplearning4j.memory.BenchmarkCnnMemory;
import org.deeplearning4j.memory.MemoryTest;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.nn.conf.WorkspaceMode;

public class StandardPerfBenchmarks {

    public static void main(String[] args) throws Exception {

        int testNum = 2;

        ModelType modelType;
        int[] batchSizes;
        MemoryTest memoryTest;
        WorkspaceMode workspaceMode;

        switch (testNum){
            //MultiLayerNetwork tests:
            case 0:
                modelType = ModelType.ALEXNET;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 1:
                modelType = ModelType.ALEXNET;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 2:
                modelType = ModelType.VGG16;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 3:
                modelType = ModelType.VGG16;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;


            //ComputationGraph tests:
            case 4:
                modelType = ModelType.GOOGLELENET;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 5:
                modelType = ModelType.GOOGLELENET;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 6:
                modelType = ModelType.INCEPTIONRESNETV1;
                memoryTest = MemoryTest.INFERENCE;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;
            case 7:
                modelType = ModelType.INCEPTIONRESNETV1;
                memoryTest = MemoryTest.TRAINING;
                batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64};
                workspaceMode = WorkspaceMode.SINGLE;
                break;

            default:
                throw new IllegalArgumentException("Invalid test: " + testNum);
        }

        BenchmarkCnn.main(new String[]{
                "--modelType", modelType.toString(),
//                "--batchSizes", Arrays.toString(batchSizes),
                "--memoryTest", memoryTest.toString(),
                "--workspaceMode", workspaceMode.toString()
        });

    }

}
