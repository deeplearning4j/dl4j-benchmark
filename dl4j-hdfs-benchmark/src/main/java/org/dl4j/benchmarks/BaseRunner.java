package org.dl4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.concurrent.atomic.AtomicLong;


/**
 *
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class BaseRunner {
    @Parameter(names = {"-p"}, description = "HDFS path where data is stored", required = false)
    protected String pathHdfs = null;

    @Parameter(names = {"-k"}, description = "Local path where data is stored", required = false)
    protected String pathLocal = null;

    @Parameter(names = {"-d"}, description = "Delay (in ms) between fetches", required = false)
    protected long computationalDelay = 0;

    @Parameter(names = {"-c"}, description = "Create datasets", required = false, arity = 1)
    protected boolean create = false;

    @Parameter(names = {"-n"}, description = "Number of batches to create", required = false)
    protected int numBatches = 1000;

    public void run(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        computationalDelay = Math.abs(computationalDelay);
        AtomicLong counter = new AtomicLong(0);
        long timeStart = 0L;
        long timeStop = 0L;



        if (pathHdfs == null && pathLocal == null)
            throw new IllegalStateException("Both HDFS and local paths are undefined. Nothing to benchmark here.");
        else if (pathHdfs != null) {
            // checking out remote throughput
        } else if (pathLocal != null) {
            // checking out local throughput

            if (create) {
                INDArray features = Nd4j.rand(119, 128, 3, 224, 224);
                INDArray labels = Nd4j.create(128, 1000).getColumn(0).assign(1.0);

                DataSet ds = new DataSet(features, labels);
                for (int e = 0; e < numBatches; e++) {
                    File t = new File(pathLocal, "temp_" +e + ".bin");
                    ds.save(t);
                }
            }

            ExistingMiniBatchDataSetIterator iterator = new ExistingMiniBatchDataSetIterator(new File(pathLocal));

            timeStart = System.currentTimeMillis();
            if (computationalDelay == 0) {
                while (iterator.hasNext()) {
                    DataSet ds = iterator.next();
                    counter.addAndGet(ds.getMemoryFootprint());
                }
            } else {
                AsyncDataSetIterator adsi = new AsyncDataSetIterator(iterator, 3, true);
                while (adsi.hasNext()) {
                    DataSet ds = adsi.next();
                    counter.addAndGet(ds.getMemoryFootprint());
                    Thread.sleep(computationalDelay);
                }
                adsi.shutdown();
            }
            timeStop = System.currentTimeMillis();
        }

        double seconds = (double) (timeStop - timeStart) / 1000.0;
        log.info("Results:");
        log.info("Time spent: {} ms", timeStop - timeStart);
        log.info("Bytes read: {}", counter.get());
        log.info("Throughput: {} MB/s", String.format("%.2f", ((counter.get() / seconds) / 1024 / 1024)));
    }

    public static void main( String[] args ) throws Exception{
        new BaseRunner().run(args);
    }
}
