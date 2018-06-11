package org.deeplearning4j.simple;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TempPhysicalBytes {

    public static void main(String[] args) {
        Nd4j.create(1);
        System.out.println("Physical bytes: " + Pointer.physicalBytes());

        INDArray arr = Nd4j.create(1_000_000_000);

        System.out.println("Physical bytes: " + Pointer.physicalBytes());
    }

}
