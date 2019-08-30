package org.deeplearning4j.mkldnn;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.adapters.OutputAdapter;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jCpu;

public class BenchmarkMklDnnConv2dOpContext {

    public static void main(String[] args) {
        Nd4j.getExecutioner().enableVerboseMode(true);

        //Disable MKL-DNN and use convolution layer inference:
        Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);

        INDArray in = Nd4j.rand(new int[]{8,32,64,64});

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new ConvolutionLayer.Builder().nIn(32).nOut(32).kernelSize(2,2).stride(1,1)
                        .convolutionMode(ConvolutionMode.Same).activation(Activation.IDENTITY).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //Use output adaptor to avoid out of workspace allocations
        OutputAdapter<Void> outputAdapter = new OutputAdapter<Void>(){
            @Override
            public Void apply(INDArray... outputs) { return null; }};

        for(int i=0; i<10; i++ ){
            net.output(in, null, null, outputAdapter);
        }

        int n = 100;
        long start = System.currentTimeMillis();
        for(int i=0; i<n; i++ ){
            net.output(in, null, null, outputAdapter);
        }
        long end = System.currentTimeMillis();
        double avg = (end-start) / (double)n;
        System.out.println("DL4J ConvolutionLayer: average " + avg + " ms");



        //Enable MKL-DNN and use conv2d op
        for(boolean useMklDnn : new boolean[]{false, true}) {
//        for(boolean useMklDnn : new boolean[]{true}) {
            Nd4jCpu.Environment.getInstance().setUseMKLDNN(useMklDnn);
            INDArray w = net.getParam("0_W").permute(2, 3, 1, 0).dup();  //From [oD, iD, kH, kW] to [kH, kW, iC, oC]
            INDArray b = net.getParam("0_b").reshape(32).dup();
            INDArray out = in.like();

            OpContext c = Nd4j.getExecutioner().buildContext();
            c.setIArguments(2, 2,    //Kernel
                    1, 1,    //Stride
                    0, 0,    //Padding
                    1, 1,    //Dilation
                    1,      //SAME
                    0);       //NCHW
            c.setInputArray(0, in);
            c.setInputArray(1, w);
            c.setInputArray(2, b);
            c.setOutputArray(0, out);

            Conv2D conv2D = new Conv2D();

            Nd4j.exec(conv2D, c);

            start = System.currentTimeMillis();
            for (int i = 0; i < n; i++) {
                Nd4j.exec(conv2D, c);
            }
            end = System.currentTimeMillis();
            avg = (end - start) / (double) n;
            System.out.println("conv2d op (use mkl=" + useMklDnn + "): average " + avg + " ms");
        }
    }
}
