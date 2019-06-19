package org.nd4j;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
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
        shapeDim.add(new Pair<>(new long[]{100}, null));
        shapeDim.add(new Pair<>(new long[]{100}, new int[]{0}));
        shapeDim.add(new Pair<>(new long[]{32,1024}, new int[]{1}));
        shapeDim.add(new Pair<>(new long[]{16384,256}, new int[]{0}));
        shapeDim.add(new Pair<>(new long[]{256,16384}, new int[]{0}));
        shapeDim.add(new Pair<>(new long[]{32,128,256,256}, new int[]{2,3}));
        shapeDim.add(new Pair<>(new long[]{32,128,256,256}, null));
        shapeDim.add(new Pair<>(new long[]{32,512,16,16}, new int[]{2,3}));



        for(boolean warmup : new boolean[]{true, false}) {

            int nIter = warmup ? 20 : 100;

            for (Pair<long[],int[]> test : shapeDim) {
                INDArray arr = Nd4j.create(test.getFirst());
                INDArray arr2 = arr.dup();

                int[] dims = test.getSecond();

                //SUM
                System.gc();
                long startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    if(dims == null){
                        arr.sumNumber();
                    } else {
                        arr.sum(dims);
                    }
                }
                long endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / (double)nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) +
                                    (dims == null ? ".sumNumber()" : ".sum(" + Arrays.toString(dims) + ")") +
                                    " in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }

                //VAR
                System.gc();
                startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    if(dims == null){
                        arr.varNumber();
                    } else {
                        arr.var(dims);
                    }
                }
                endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / (double)nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) +
                            (dims == null ? ".varNumber()" : ".var(" + Arrays.toString(dims) + ")") +
                            " in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }


                //MEAN
                System.gc();
                startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    if(dims == null){
                        arr.meanNumber();
                    } else {
                        arr.mean(dims);
                    }
                }
                endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / (double)nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".mean(" + Arrays.toString(dims)
                            + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }

                //ASSIGN
                System.gc();
                startNano = System.nanoTime();
                for (int i = 0; i < nIter; i++) {
                    arr.assign(arr2);
                }
                endNano = System.nanoTime();

                if(!warmup) {
                    double avg = (endNano - startNano) / (double)nIter;
                    log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".assign(" + Arrays.toString(test.getFirst())
                            + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                }

                //ADDIROWVECTOR, ADDICOLUMNVECTOR
                if(test.getFirst().length == 2){
                    System.gc();
                    startNano = System.nanoTime();
                    INDArray row = Nd4j.create(test.getFirst()[1]);
                    for (int i = 0; i < nIter; i++) {
                        arr.addiRowVector(row);
                    }
                    endNano = System.nanoTime();

                    if(!warmup) {
                        double avg = (endNano - startNano) / (double)nIter;
                        log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".addiRowVector(" + Arrays.toString(row.shape())
                                + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                    }

                    System.gc();
                    startNano = System.nanoTime();
                    INDArray col = Nd4j.create(test.getFirst()[0]);
                    for (int i = 0; i < nIter; i++) {
                        arr.addiColumnVector(col);
                    }
                    endNano = System.nanoTime();

                    if(!warmup) {
                        double avg = (endNano - startNano) / (double)nIter;
                        log.info("Completed " + nIter + " iterations of " + Arrays.toString(test.getFirst()) + ".addiColumnVector(" + Arrays.toString(row.shape())
                                + ") in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                    }
                }

                //Add on dimensions [1], [2], [3], [1,2], [1,3], [1,2,3]
                if(test.getFirst().length == 4){
                    for( int dim = 1; dim <= 3; dim++ ){
                        System.gc();
                        startNano = System.nanoTime();
                        INDArray vector = Nd4j.create(test.getFirst()[dim]);
                        for (int i = 0; i < nIter; i++) {
                            Broadcast.add(arr, vector, arr, dim);
                        }
                        endNano = System.nanoTime();

                        if(!warmup) {
                            double avg = (endNano - startNano) / (double)nIter;
                            log.info("Completed " + nIter + " iterations of Broadcast.add(" + Arrays.toString(test.getFirst()) + "," + Arrays.toString(vector.shape())
                                    + ",dim=[" + dim + "] in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                        }
                    }

                    for(int[] dim : new int[][]{{2,3}, {1,3}}) {
                        System.gc();
                        startNano = System.nanoTime();
                        long[] smallerShape = new long[]{test.getFirst()[dim[0]], test.getFirst()[dim[1]]};
                        INDArray smaller = Nd4j.create(smallerShape);
                        for (int i = 0; i < nIter; i++) {
                            Broadcast.add(arr, smaller, arr, dim);
                        }
                        endNano = System.nanoTime();

                        if (!warmup) {
                            double avg = (endNano - startNano) / (double) nIter;
                            log.info("Completed " + nIter + " iterations of Broadcast.add(" + Arrays.toString(test.getFirst()) + "," + Arrays.toString(smaller.shape())
                                    + ",dim=" + Arrays.toString(dim) + " in " + (endNano - startNano) + "ns - average " + formatNanos(avg) + " per iteration");
                        }
                    }
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
