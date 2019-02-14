package org.nd4j;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class SmallArrayBenchmark {

    public static void main(String[] args) {

        val wsconf = WorkspaceConfiguration.builder()
                .initialSize(10 * 1024L * 1024L)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .build();

        val array = Nd4j.create(DataType.FLOAT, 100);
        for (int e = 0; e < 1000; e++) {
            try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsconf, "kjmnf,ndsfhnsdjhflljkl131334")) {
                array.sumNumber();
            }
        }

        int iterations = 1000000;
        val timeStart = System.nanoTime();
        for (int e = 0; e < iterations; e++) {
            try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsconf, "kjmnf,ndsfhnsdjhflljkl131334")) {
                array.sumNumber();
            }
        }
        val timeEnd = System.nanoTime();
        log.info("Average time: {} us", (timeEnd - timeStart) / (double) iterations / (double) 1000);
    }

}
