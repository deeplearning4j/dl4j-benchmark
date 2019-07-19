package ai.skymind;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;

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


    public static void startMemoryLoggingThread(final long msFreq){
        Nd4j.create(1);

        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                while(true){
                    try{
                        Thread.sleep(msFreq);;
                    } catch (InterruptedException e){ }
                    long b = Pointer.totalBytes();
                    long pb = Pointer.physicalBytes();
                    log.info("JavaCPP Memory: {} total, {} physical", b, pb);
                }
            }
        });
        t.setDaemon(true);
        t.start();
    }

}
