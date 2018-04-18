package org.deeplearning4j.simple;

import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

public class SimpleCol2ImBenchmark {

    public static void main(String[] args) {

        int[] inputShape = new int[]{16, 3, 224, 224};
        INDArray input = Nd4j.create(inputShape);

        //First im2col call (first AlexNet layer)
        int miniBatch = 16;
        int inH = 224;
        int inW = 224;
        int outH = 55;
        int outW = 55;
        int inDepth = 3;
        int kH = 11;
        int kW = 11;
        int sx = 4;
        int sy = 4;
        int ph = 2;
        int pw = 2;

        INDArray eps6d = Nd4j.create(kW, kH, inDepth, outW, outH, miniBatch);
        eps6d = eps6d.permute(5, 2, 1, 0, 4, 3);
        INDArray epsNext = Nd4j.create(inDepth, miniBatch, inH, inW).permute(1, 0, 2, 3);

        int[] expEps6dShape = new int[]{16, 3, 11, 11, 55, 55};
        if(!Arrays.equals(expEps6dShape, eps6d.shape())){
            throw new RuntimeException();
        }

        int[] epsNextShape = new int[]{16, 3, 224, 224};
        if(!Arrays.equals(epsNextShape, epsNext.shape())){
            throw new RuntimeException();
        }

        int nIter = 100;

        long start = System.currentTimeMillis();
        for (int i = 0; i < nIter; i++) {
            Convolution.col2im(eps6d, epsNext, sy, sx, ph, pw, inH, inW, 1, 1);
//            Convolution.col2im(eps6d, epsNext, sy, sx, ph, pw, inH, inW);
        }
        long end = System.currentTimeMillis();

        double avg = (end - start) / (double) nIter;
        System.out.println("Average col2im: " + avg);
    }
}
