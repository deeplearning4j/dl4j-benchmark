package ai.skymind.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;

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

}
