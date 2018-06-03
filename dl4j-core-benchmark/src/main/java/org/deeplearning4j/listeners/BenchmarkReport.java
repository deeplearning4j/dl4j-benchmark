package org.deeplearning4j.listeners;

import lombok.Data;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicDouble;
import oshi.SystemInfo;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.OperatingSystem;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Reporting for BenchmarkListener.
 *
 * @author Justin Long (@crockpotveggies)
 */
@Data
public class BenchmarkReport {

    private String name;
    private String description;
    private List<String> devices = new ArrayList<>();
    private String backend;
    private String cpuCores;
    private String blasVendor;
    private String modelSummary;
    private String cudaVersion;
    private String cudnnVersion;
    private boolean periodicGCEnabled;
    private int periodicGCFreq;
    private int occasionalGCFreq;
    private boolean isParallelWrapper;
    private int parallelWrapperNumThreads;
    private int numParams;
    private int numLayers;
    //Next 4: updated by benchmark listener (possibly by multiple threads with PW)
    private Map<Long, AtomicLong> iterationsPerThread = new ConcurrentHashMap<>();
    private Map<Long, AtomicLong> totalIterationTimePerThread = new ConcurrentHashMap<>();
    private Map<Long, AtomicDouble> totalSamplesSecPerThread = new ConcurrentHashMap<>();
    private Map<Long, AtomicDouble> totalBatchesSecPerThread = new ConcurrentHashMap<>();
    //Next 4: only collected if NOT using PW
    private double avgFeedForward;
    private double avgBackprop;
    private double avgFit;
    private double avgUpdater;
    private int batchSize;

    public BenchmarkReport(String name, String description) {
        this.name = name;
        this.description = description;

        Properties env = Nd4j.getExecutioner().getEnvironmentInformation();

        backend = env.get("backend").toString();
        cpuCores = env.get("cores").toString();
        blasVendor = env.get("blas.vendor").toString();

        // if CUDA is present, add GPU information
        try {
            List devicesList = (List) env.get("cuda.devicesInformation");
            Iterator deviceIter = devicesList.iterator();
            while (deviceIter.hasNext()) {
                Map dev = (Map) deviceIter.next();
                devices.add(dev.get("cuda.deviceName") + " " + dev.get("cuda.deviceMajor") + " " + dev.get("cuda.deviceMinor") + " " + dev.get("cuda.totalMemory"));
            }
        } catch (Throwable e) {
            SystemInfo sys = new SystemInfo();
            devices.add(sys.getHardware().getProcessor().getName());
        }

        // also get CUDA version
        try {
            Field f = Class.forName("org.bytedeco.javacpp.cuda").getField("__CUDA_API_VERSION");
            int version = f.getInt(null);
            this.cudaVersion = Integer.toString(version);
        } catch (Throwable e) {
            this.cudaVersion = "n/a";
        }

        // if cuDNN is present, let's get that info
        try {
            Method m = Class.forName("org.bytedeco.javacpp.cudnn").getDeclaredMethod("cudnnGetVersion");
            long version = (long) m.invoke(null);
            this.cudnnVersion = Long.toString(version);
        } catch (Throwable e) {
            this.cudnnVersion = "n/a";
        }
    }

    public void setModel(Model model) {
        this.numParams = model.numParams();

        if (model instanceof MultiLayerNetwork) {
            this.modelSummary = ((MultiLayerNetwork) model).summary();
            this.numLayers = ((MultiLayerNetwork) model).getnLayers();
        }
        if (model instanceof ComputationGraph) {
            this.modelSummary = ((ComputationGraph) model).summary();
            this.numLayers = ((ComputationGraph) model).getNumLayers();
        }
    }

    public void addIterations(long threadId, long iterations) {
        this.iterationsPerThread.computeIfAbsent(threadId, AtomicLong::new)
                .addAndGet(iterations);
    }

    public void addIterationTime(long threadId, long iterationTime) {
        this.totalIterationTimePerThread.computeIfAbsent(threadId, AtomicLong::new)
                .addAndGet(iterationTime);
    }

    public void addSamplesSec(long threadId, double samplesSec) {
        this.totalSamplesSecPerThread.computeIfAbsent(threadId, AtomicDouble::new)
                .addAndGet(samplesSec);
    }

    public void addBatchesSec(long threadId, double batchesSec) {
        this.totalBatchesSecPerThread.computeIfAbsent(threadId, AtomicDouble::new)
                .addAndGet(batchesSec);
    }

    public List<String> devices() {
        return devices;
    }

    /**
     * @return Average iteration time per batch for a single executor - averaged over all threads
     */
    public double avgIterationTime() {
        return avgOverThreads(totalIterationTimePerThread, iterationsPerThread);
    }

    /**
     * @return Average samples per second - on a *per executor* basis. To get the combined number of batches per
     * sec for all executors, use {@link #avgSamplesPerSecCombined()}
     */
    public double avgSamplesSecPerExecutor() {
        return avgOverThreads(totalIterationTimePerThread, iterationsPerThread);
    }

    /**
     * sum_threads (avg samples per sec for each thread)
     *
     * @return Average combined samples per sec
     */
    public double avgSamplesPerSecCombined() {
        //NOTE: THIS ASSUMES ALL THREADS WERE RUNNING CONCURRENTLY.
        //If some threads were shut down, and new threads added later - this would overestimate the total throughput!
        return sumOverThreadsD(totalSamplesSecPerThread, iterationsPerThread);
    }

    public double avgBatchesSecPerExecutor() {
        return avgOverThreadsD(totalBatchesSecPerThread, iterationsPerThread);
    }

    public double avgBatchesPerSecCombined() {
        return sumOverThreadsD(totalBatchesSecPerThread, iterationsPerThread);
    }

    public double avgFeedForward() {
        return avgFeedForward;
    }

    public double avgBackprop() {
        return avgBackprop;
    }

    public String getModelSummary() {
        return modelSummary;
    }

    private long sumAllThreads(Map<Long, AtomicLong> map) {
        long result = 0;
        for (AtomicLong entry : map.values()) {
            result += entry.get();
        }
        return result;
    }

    private double sumAllThreadsD(Map<Long, AtomicDouble> map) {
        double result = 0.0;
        for (AtomicDouble entry : map.values()) {
            result += entry.get();
        }
        return result;
    }

    /**
     * Returns: sum_threads (numerator.get(threadId) / denominator.get(threadId))
     */
    private double sumOverThreadsD(Map<Long, AtomicDouble> numerator, Map<Long, AtomicLong> denominator) {
        double sumOverThreads = 0.0;
        for (Map.Entry<Long, AtomicLong> e : denominator.entrySet()) {
            long denominatorThisThread = e.getValue().get();
            double numeratorThisThread = numerator.get(e.getKey()).get();
            sumOverThreads += numeratorThisThread / (double) denominatorThisThread;
        }
        return sumOverThreads;
    }

    /**
     * Returns: sum_threads(numerator.get(threadId)) / sum_threads(denominator.get(threadId))
     */
    private double avgOverThreadsD(Map<Long, AtomicDouble> numerator, Map<Long, AtomicLong> denominator) {
        double sumNumerator = sumAllThreadsD(numerator);
        long sumDenominator = sumAllThreads(denominator);
        return sumNumerator / sumDenominator;
    }

    /**
     * Returns: sum_threads(numerator.get(threadId)) / sum_threads(denominator.get(threadId))
     */
    private double avgOverThreads(Map<Long, AtomicLong> numerator, Map<Long, AtomicLong> denominator) {
        long sumNumerator = sumAllThreads(numerator);
        long sumDenominator = sumAllThreads(denominator);
        return sumNumerator / (double) sumDenominator;
    }


    public String toString() {
        DecimalFormat df = new DecimalFormat("#.##");

        SystemInfo sys = new SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        HardwareAbstractionLayer hardware = sys.getHardware();


        List<String[]> table = new ArrayList<>();
        table.add(new String[]{"Name", name});
        table.add(new String[]{"Description", description});
        table.add(new String[]{"Operating System",
                os.getManufacturer() + " " +
                        os.getFamily() + " " +
                        os.getVersion().getVersion()});
        table.add(new String[]{"Devices", devices().get(0)});
        table.add(new String[]{"CPU Cores", cpuCores});
        table.add(new String[]{"Backend", backend});
        table.add(new String[]{"BLAS Vendor", blasVendor});
        table.add(new String[]{"CUDA Version", cudaVersion});
        table.add(new String[]{"CUDNN Version", cudnnVersion});
        table.add(new String[]{"Periodic GC enabled", String.valueOf(periodicGCEnabled)});
        if (periodicGCEnabled) {
            table.add(new String[]{"Periodic GC frequency", String.valueOf(periodicGCFreq)});
        }
        table.add(new String[]{"Occasional GC Freq", String.valueOf(occasionalGCFreq)});
        table.add(new String[]{"Parallel Wrapper", String.valueOf(isParallelWrapper)});
        if(isParallelWrapper){
            table.add(new String[]{"Parallel Wrapper # threads", String.valueOf(parallelWrapperNumThreads)});
        }
        table.add(new String[]{"Total Params", Integer.toString(numParams)});
        table.add(new String[]{"Total Layers", Integer.toString(numLayers)});
        if(!isParallelWrapper) {
            table.add(new String[]{"Avg Feedforward (ms)", df.format(avgFeedForward)});
            table.add(new String[]{"Avg Backprop (ms)", df.format(avgBackprop)});
            table.add(new String[]{"Avg Fit (ms)", df.format(avgFit)});
            table.add(new String[]{"Avg Iteration (ms)", df.format(avgIterationTime())});
            table.add(new String[]{"Avg Samples/sec", df.format(avgSamplesSecPerExecutor())});
            table.add(new String[]{"Avg Batches/sec", df.format(avgBatchesSecPerExecutor())});
        } else {
            table.add(new String[]{"Avg Samples/sec (per executor)", df.format(avgSamplesSecPerExecutor())});
            table.add(new String[]{"Avg Samples/sec (total)", df.format(avgSamplesPerSecCombined())});
            table.add(new String[]{"Avg Batches/sec (per executor)", df.format(avgBatchesSecPerExecutor())});
            table.add(new String[]{"Avg Batches/sec (total)", df.format(avgBatchesPerSecCombined())});
        }
        table.add(new String[]{"Batch size", Integer.toString(batchSize)});

        StringBuilder sb = new StringBuilder();

        for (final Object[] row : table) {
            sb.append(String.format("%28s %45s\n", row));
        }

        return sb.toString();
    }

}
