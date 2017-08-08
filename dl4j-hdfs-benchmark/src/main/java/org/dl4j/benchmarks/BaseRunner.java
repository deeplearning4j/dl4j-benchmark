package org.dl4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;


import java.io.BufferedInputStream;
import java.io.File;
import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;


/**
 *
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class BaseRunner {
    @Parameter(names = {"-h"}, description = "HDFS path where data is stored", required = false)
    protected String pathHdfs = null;

    @Parameter(names = {"-l"}, description = "Local path where data is stored", required = false)
    protected String pathLocal = null;

    @Parameter(names = {"-d"}, description = "Delay (in ms) between fetches", required = false)
    protected long computationalDelay = 1;

    @Parameter(names = {"-c"}, description = "Create datasets", required = false, arity = 1)
    protected boolean create = false;

    @Parameter(names = {"-n"}, description = "Number of batches to create", required = false)
    protected int numBatches = 500;

    @Parameter(names = {"--host"}, description = "HDFS host name")
    protected String hdfsHost = null;

    @Parameter(names = {"--port"}, description = "HDFS port")
    protected int hdfsPort = 0;

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
        AtomicLong timer = new AtomicLong(0);
        List<Long> times = new ArrayList<>();
        long timeStart = 0L;
        long timeStop = 0L;
        int cnt = 0;


        if (pathHdfs != null && pathLocal != null)
            throw new IllegalStateException("Both HDFS and local paths undefined. Please pick one at a time");
        else if (pathHdfs == null && pathLocal == null)
            throw new IllegalStateException("Both HDFS and local paths are undefined. Nothing to benchmark here.");
        else if (pathHdfs != null) {
            // checking out remote throughput
            Configuration conf = new Configuration();
            String hdfsBox = "hdfs://" + hdfsHost + (hdfsPort == 0 ? "" : ":" + hdfsPort);
            conf.set("fs.default.name", hdfsBox);

            FileSystem fs = FileSystem.get(conf);

            if (create) {
                log.info("Creating datasets @ HDFS...");

                INDArray features = Nd4j.rand(119, 32, 3, 224, 224);
                INDArray labels = Nd4j.create(32, 1000).getColumn(0).assign(1.0);

                DataSet ds = new DataSet(features, labels);
                for (int e = 0; e < numBatches; e++) {
                    String path = pathHdfs + "/" + ("dataset-" +e + ".bin");
                    URI uri = new URI(hdfsBox + "/" + pathHdfs + "/" + path);
                    ds.save(fs.append(new Path(uri)).getWrappedStream());
                }
            }

            log.info("Starting HDFS benchmarking...");

            URI uri = new URI(hdfsBox + "/" + pathHdfs);
            RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(new Path(uri), false);
            timeStart = System.currentTimeMillis();
            while (iterator.hasNext()) {
                LocatedFileStatus lfs = iterator.next();
                Path path = lfs.getPath();

                DataSet ds = new DataSet();
                try (InputStream is = fs.open(path).getWrappedStream(); BufferedInputStream bis = new BufferedInputStream(is)) {
                    ds.load(bis);
                }

                timeStop = System.currentTimeMillis();
                counter.addAndGet(ds.getMemoryFootprint());

                timer.addAndGet(timeStop - timeStart);
                times.add(timeStop - timeStart);

                cnt++;
                if (cnt % 100 == 0)
                    log.info("{} datasets processed", cnt);

                timeStart = System.currentTimeMillis();
            }

        } else if (pathLocal != null) {
            // checking out local throughput

            if (create) {
                log.info("Creating datasets...");

                INDArray features = Nd4j.rand(119, 32, 3, 224, 224);
                INDArray labels = Nd4j.create(32, 1000).getColumn(0).assign(1.0);

                DataSet ds = new DataSet(features, labels);
                for (int e = 0; e < numBatches; e++) {
                    File t = new File(pathLocal, "dataset-" +e + ".bin");
                    ds.save(t);
                }
            }

            log.info("Starting local benchmarking...");

            ExistingMiniBatchDataSetIterator iterator = new ExistingMiniBatchDataSetIterator(new File(pathLocal));

            timeStart = System.currentTimeMillis();
            if (computationalDelay == 0) {
                while (iterator.hasNext()) {
                    DataSet ds = iterator.next();
                    timeStop = System.currentTimeMillis();
                    counter.addAndGet(ds.getMemoryFootprint());

                    timer.addAndGet(timeStop - timeStart);
                    times.add(timeStop - timeStart);

                    cnt++;
                    if (cnt % 100 == 0)
                        log.info("{} datasets processed", cnt);

                    timeStart = System.currentTimeMillis();
                }
            } else {
                AsyncDataSetIterator adsi = new AsyncDataSetIterator(iterator, 3, true);

                timeStart = System.currentTimeMillis();
                while (adsi.hasNext()) {
                    DataSet ds = adsi.next();
                    timeStop = System.currentTimeMillis();
                    counter.addAndGet(ds.getMemoryFootprint());
                    if (computationalDelay > 1)
                        Thread.sleep(computationalDelay);

                    timer.addAndGet(timeStop - timeStart);
                    times.add(timeStop - timeStart);

                    cnt++;
                    if (cnt % 100 == 0)
                        log.info("{} datasets processed", cnt);

                    timeStart = System.currentTimeMillis();
                }
                adsi.shutdown();
            }

            Collections.sort(times);
        }

        double seconds = (double) (timer.get()) / 1000.0;
        log.info("Results:");
        log.info("Median time: {} ms", times.get(times.size() / 2));
        log.info("Time spent: {} ms", timer.get());
        log.info("Bytes read: {}", counter.get());
        log.info("Throughput: {} MB/s", String.format("%.2f", ((counter.get() / seconds) / 1024 / 1024)));
    }

    public static void main( String[] args ) throws Exception{
        new BaseRunner().run(args);
    }
}
