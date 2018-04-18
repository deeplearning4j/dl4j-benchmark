package org.deeplearning4j.simple;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

public class SimpleIm2ColBenchmark {

    public static void main(String[] args) {

        int[] inputShape = new int[]{16, 3, 224, 224};
        INDArray input = Nd4j.create(inputShape);

        int inH = 224;
        int inW = 224;
        int miniBatch = 16;
        int outH = 55;
        int outW = 55;
        int inDepth = 3;
        int kH = 11;
        int kW = 11;
        int sx = 4;
        int sy = 4;
        int ph = 2;
        int pw = 2;
        boolean isSameMode = false;

        INDArray col = Nd4j.createUninitialized(new int[]{miniBatch, outH, outW, inDepth, kH, kW}, 'c');
        INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);


        int nIter = 100;

        long start = System.currentTimeMillis();
        for (int i = 0; i < nIter; i++) {
            Convolution.im2col(input, kH, kW, sy, sx, ph, pw, isSameMode, col2);
        }
        long end = System.currentTimeMillis();

        double avg = (end - start) / (double) nIter;
        System.out.println("Average im2col: " + avg);
    }
}
