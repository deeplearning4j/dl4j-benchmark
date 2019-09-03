package org.deeplearning4j.layers;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jCpu;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ConvBackpropBenchmarks {

    private static final int WARMUP = 30;
    private static final int ITERS = 100;

    @Builder
    public static class TestCase {
        private String test;
        private long[] inSize;
        private long[] outSize;
        private int[] k;
        private int[] s;
        private int[] p;
        private boolean same;

        @Override
        public String toString() {
            return test + " - in=" + Arrays.toString(inSize) + ", outSize=" + Arrays.toString(outSize) + ", k=" + Arrays.toString(k)
            + ", s=" + Arrays.toString(s) + ", p=" + Arrays.toString(p) + ", same=" + same;
        }
    }


    public static void main(String[] args) {
        int warmup = 20;
        int runs = 100;

        int mb = 32;

        List<TestCase> l = new ArrayList<>();

        //VGG16 test cases:
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 512, 14, 14}).outSize(new long[]{mb, 512, 7, 7})
                .k(new int[]{2,2}).s(new int[]{2, 2}).p(new int[]{0, 0}).same(false).build());
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 512, 28, 28}).outSize(new long[]{mb, 512, 14, 14})
                .k(new int[]{2,2}).s(new int[]{2, 2}).p(new int[]{0, 0}).same(false).build());
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 256, 56, 56}).outSize(new long[]{mb, 256, 28, 28})
                .k(new int[]{2,2}).s(new int[]{2, 2}).p(new int[]{0, 0}).same(false).build());
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 128, 112, 112}).outSize(new long[]{mb, 128, 56, 56})
                .k(new int[]{2,2}).s(new int[]{2, 2}).p(new int[]{0, 0}).same(false).build());
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 64, 224, 224}).outSize(new long[]{mb, 64, 112, 112})
                .k(new int[]{2,2}).s(new int[]{2, 2}).p(new int[]{0, 0}).same(false).build());

        //Resnet50 test case:
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 2048, 7, 7}).outSize(new long[]{mb, 2048, 1, 1})
                .k(new int[]{7, 7}).s(new int[]{7, 7}).p(new int[]{0, 0}).same(false).build());
        l.add(TestCase.builder().test("maxpool")
                .inSize(new long[]{mb, 64, 112, 112}).outSize(new long[]{mb, 64, 55, 55})
                .k(new int[]{3,3}).s(new int[]{2, 2}).p(new int[]{0, 0}).same(false).build());

        for(TestCase tc : l){

            if("maxpool".equals(tc.test)){

                INDArray in = Nd4j.create(DataType.FLOAT, tc.inSize);
                INDArray eps = Nd4j.create(DataType.FLOAT, tc.outSize);
                INDArray inGrad = Nd4j.create(DataType.FLOAT, tc.inSize);

                DynamicCustomOp op = DynamicCustomOp.builder("maxpool2d_bp")
                        .addInputs(in, eps)
                        .addOutputs(inGrad)
                        .addIntegerArguments(
                                tc.k[0], tc.k[1],
                                tc.s[0], tc.s[1],
                                tc.p[0], tc.p[1],
                                1, 1,   //Dilation
                                tc.same ? 1 : 0,
                                0)      //NCHW
                        .build();

                for( int i=0; i<warmup; i++ ){
                    Nd4j.exec(op);
                }

                long start = System.currentTimeMillis();
                for( int i=0; i<runs; i++ ){
                    Nd4j.exec(op);
                }
                long end = System.currentTimeMillis();


                double avgMs = (end-start) / (double)runs;
                System.out.println(tc);
                System.out.println(avgMs + " ms");

            } else {
                throw new RuntimeException("Not implemented: " + tc.test);
            }

        }


    }
}
