package org.nd4j;

import org.junit.Test;

public class TestMobilenetV1 {

    @Test
    public void test() throws Exception {

        SameDiffBenchmarkRunner.main(
                "--modelClass", "org.nd4j.models.MobilenetV1",
                "--batchSize", "16",
                "--numIterWarmup", "10",
                "--numIter", "20"
        );

    }

}
