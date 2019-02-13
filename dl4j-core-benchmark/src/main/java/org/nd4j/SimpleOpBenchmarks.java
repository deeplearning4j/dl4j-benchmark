package org.nd4j;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class SimpleOpBenchmarks {

    public static void main(String[] args) {

        Nd4j.getMemoryManager().togglePeriodicGc(false);

        List<Pair<long[],int[]>> shapeDim = new ArrayList<>();
        shapeDim.add(new Pair<>(new long[]{100}, new int[]{0}));
        shapeDim.add(new Pair<>(new long[]{32,1024}, new int[]{1}));
        shapeDim.add(new Pair<>(new long[]{32,128,256,256}, new int[]{2,3}));
        shapeDim.add(new Pair<>(new long[]{32,512,16,16}, new int[]{2,3}));

        int nIter = 100;

        for(boolean warmup : new boolean[]{true, false}) {
            for (Pair<long[],int[]> test : shapeDim) {
                INDArray arr = Nd4j.create(test.getFirst());
                INDArray arr2 = arr.dup();


                System.gc();
                long startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    arr.sum(test.getSecond());
                }
                long endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".sum(" + Arrays.toString(test.getSecond())
                            + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }


                System.gc();
                startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    arr.var(test.getSecond());
                }
                endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".var(" + Arrays.toString(test.getSecond())
                            + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }


                System.gc();
                startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    arr.mean(test.getSecond());
                }
                endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".mean(" + Arrays.toString(test.getSecond())
                            + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }

                System.gc();
                startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    arr.assign(arr2);
                }
                endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".assign(" + Arrays.toString(test.getFirst())
                            + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }
            }
        }
    }

    private static final DecimalFormat df = new DecimalFormat("#0.00");

    public static String formatNanos(double d){
        if(d >= 1e9){
            //Seconds
            return df.format(d / 1e9) + " sec";
        } else if(d >= 1e6 ){
            //ms
            return df.format(d / 1e6) + " ms";
        } else if(d >= 1e3 ){
            //us
            return df.format(d / 1e3) + " us";
        } else {
            //ns
            return df.format(d) + " ns";
        }
    }

}
