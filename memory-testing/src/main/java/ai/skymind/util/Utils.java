package ai.skymind.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.normalization.BatchNormalization;
import org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;
import org.nd4j.nativeblas.Nd4jCpu;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class Utils {

    private Utils(){ }

    public static void logMemoryConfig(){

        long mb = Pointer.maxBytes();
        long mpb = Pointer.maxPhysicalBytes();
        long xmx = Runtime.getRuntime().maxMemory();

        log.info("JavaCPP max bytes:          {}", FileUtils.byteCountToDisplaySize(mb));
        log.info("JavaCPP max physical bytes: {}", FileUtils.byteCountToDisplaySize(mpb));
        log.info("JVM XMX:                    {}", FileUtils.byteCountToDisplaySize(xmx));
    }


    public static AtomicLong[] startMemoryLoggingThread(final long msFreq){
        Nd4j.create(1);

        final AtomicLong maxPhysBytes = new AtomicLong(Pointer.physicalBytes());
        final AtomicLong maxBytes = new AtomicLong(Pointer.totalBytes());
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                while(true){
                    try{
                        Thread.sleep(msFreq);;
                    } catch (InterruptedException e){ }
                    long b = Pointer.totalBytes();
                    long pb = Pointer.physicalBytes();
                    maxBytes.set(b);
                    maxPhysBytes.set(pb);
                    log.info("JavaCPP Memory: {} total, {} physical", b, pb);
                }
            }
        });
        t.setDaemon(true);
        t.start();

        return new AtomicLong[]{maxBytes, maxPhysBytes};
    }

    public static void disableMKLDNN(){
        //Ugly backend-independent hack for disabling mkldnn
        Properties props = Nd4j.getExecutioner().getEnvironmentInformation();
        if("cpu".equalsIgnoreCase((String)props.get("backend"))){
            try {
                Class<?> c = Class.forName("org.nd4j.nativeblas.Nd4jCpu$Environment");
                Method m = c.getDeclaredMethod("getInstance");
                Method m2 = c.getDeclaredMethod("setUseMKLDNN", boolean.class);
                Object o = m.invoke(null);
                m2.invoke(o, false);
                log.info("Disabled MKLDNN");
            } catch (Exception e){
                //OK, ignore
                e.printStackTrace();
            }
        }
    }

    public static void removeHelpers(Model m) throws Exception {
        Layer[] layers;
        if(m instanceof MultiLayerNetwork){
            layers = ((MultiLayerNetwork) m).getLayers();
        } else {
            layers = ((ComputationGraph)m).getLayers();
        }
        removeHelpers(layers);
    }

    public static void removeHelpers(Layer[] layers) throws Exception {
        for(Layer l : layers){

            if(l instanceof ConvolutionLayer){
                Field f1 = ConvolutionLayer.class.getDeclaredField("helper");
                f1.setAccessible(true);
                f1.set(l, null);
            } else if(l instanceof SubsamplingLayer){
                Field f2 = SubsamplingLayer.class.getDeclaredField("helper");
                f2.setAccessible(true);
                f2.set(l, null);
            } else if(l instanceof BatchNormalization) {
                Field f3 = BatchNormalization.class.getDeclaredField("helper");
                f3.setAccessible(true);
                f3.set(l, null);
            } else if(l instanceof LSTM){
                Field f4 = LSTM.class.getDeclaredField("helper");
                f4.setAccessible(true);
                f4.set(l, null);
            } else if(l instanceof LocalResponseNormalization){
                Field f5 = LocalResponseNormalization.class.getDeclaredField("helper");
                f5.setAccessible(true);
                f5.set(l, null);
            }


            if(l.getHelper() != null){
                throw new IllegalStateException("Did not remove helper for layer: " + l.getClass().getSimpleName());
            }
        }

        log.info("Removed helpers");
    }
}
