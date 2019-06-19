package org.nd4j;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//For profiling
@Slf4j
public class SingleOpBenchmarks {

    public static void main(String[] args) {

        Nd4j.getMemoryManager().togglePeriodicGc(false);

        List<Pair<long[],int[]>> shapeDim = new ArrayList<>();
        shapeDim.add(new Pair<>(new long[]{100}, new int[]{0}));
        shapeDim.add(new Pair<>(new long[]{32,1024}, new int[]{1}));
        shapeDim.add(new Pair<>(new long[]{32,128,256,256}, new int[]{2,3}));
        shapeDim.add(new Pair<>(new long[]{32,512,16,16}, new int[]{2,3}));

        int nIter = 200;

        int[] shape = new int[]{32,128,256,256};
        INDArray arr = Nd4j.create(shape);
        INDArray arr2 = arr.dup();


            System.gc();
            long startNano = System.nanoTime();
            for (int i = 0; i < nIter; i++) {
                arr.mean(2,3);
            }
            long endNano = System.nanoTime();

            double avg = (endNano - startNano) / nIter;
            log.info("Completed " + nIter + " iterations of " + Arrays.toString(shape) + ".mean(2,3) in " +
                    (endNano - startNano) + "ns - average " + avg + "ns per iteration");
    }

}
