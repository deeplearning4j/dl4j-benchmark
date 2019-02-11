package org.nd4j;

import org.junit.Test;

public class TestNets {

    @Test
    public void testMobilenetV1() throws Exception {
        //Small batch + iterations for faster debugging only
        SameDiffBenchmarkRunner.main(
                "--modelClass", "org.nd4j.models.MobilenetV1",
                "--batchSize", "8",
                "--numIterWarmup", "3",
                "--numIter", "5"
        );
    }

    @Test
    public void testInceptionResnetV2() throws Exception {
        //Small batch + iterations for faster debugging only
        SameDiffBenchmarkRunner.main(
                "--modelClass", "org.nd4j.models.InceptionResnetV2",
                "--batchSize", "8",
                "--numIterWarmup", "3",
                "--numIter", "5"
        );
    }

    @Test
    public void testResnetV2() throws Exception {
        //Small batch + iterations for faster debugging only
        SameDiffBenchmarkRunner.main(
                "--modelClass", "org.nd4j.models.ResnetV2",
                "--batchSize", "8",
                "--numIterWarmup", "3",
                "--numIter", "5"
        );
    }
}
