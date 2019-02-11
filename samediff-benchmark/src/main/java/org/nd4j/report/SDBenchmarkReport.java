package org.nd4j.report;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import lombok.Data;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicDouble;
import org.nd4j.util.StringUtils;
import org.nd4j.versioncheck.VersionCheck;
import org.nd4j.versioncheck.VersionInfo;
import oshi.SystemInfo;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.OperatingSystem;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Reporting for BenchmarkListener.
 *
 * @author Justin Long (@crockpotveggies)
 */
@Data
public class SDBenchmarkReport {

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
    private long numParams;
    private int numLayers;
    private int batchSize;
    //Benchmark stats
    private LongArrayList fwdPassTimes = new LongArrayList();
    private LongArrayList gradCalcTimes = new LongArrayList();
    private LongArrayList fitTimes = new LongArrayList();



    public SDBenchmarkReport(String name, String description) {
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

    public void addForwardTimeMs(long fwdMS){
        fwdPassTimes.add(fwdMS);
    }

    public void addGradientCalcTime(long bwdMS){
        gradCalcTimes.add(bwdMS);
    }

    public void addGradientFitTime(long fitMS){
        fitTimes.add(fitMS);
    }

    public List<String> devices() {
        return devices;
    }

    public double avgForwardTime(){
        return average(fwdPassTimes);
    }

    public double stdForwardTime(){
        return std(fwdPassTimes);
    }

    public double avgBackpropTime(){
        return average(gradCalcTimes);
    }

    public double stdBackpropTime(){
        return std(gradCalcTimes);
    }

    public double avgFitTime(){
        return average(fitTimes);
    }

    public double stdFitTime(){
        return std(fitTimes);
    }

    protected double average(LongArrayList arrayList){
        long[] arr = arrayList.toLongArray();
        double sum = 0.0;
        for(long l : arr){
            sum += l;
        }
        return sum / arr.length;
    }

    protected double std(LongArrayList arrayList){
        long[] arr = arrayList.toLongArray();
        double std = Nd4j.createFromArray(arr).castTo(DataType.DOUBLE).stdNumber().doubleValue();
        return std;
    }


    public static String inferVersion(){
        List<VersionInfo> vi = VersionCheck.getVersionInfos();

        for(VersionInfo v : vi){
            if("org.deeplearning4j".equals(v.getGroupId()) && "deeplearning4j-core".equals(v.getArtifactId())){
                String version = v.getBuildVersion();
                if(version.contains("SNAPSHOT")){
                    return version + " (" + v.getCommitIdAbbrev() + ")";
                }
                return version;
            }
        }

        return " (could not infer version)";
    }

    public String toString() {


        DecimalFormat df = new DecimalFormat("#.##");

        SystemInfo sys = new SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        HardwareAbstractionLayer hardware = sys.getHardware();
        String procName = sys.getHardware().getProcessor().getName();
        long totalMem = sys.getHardware().getMemory().getTotal();

        long xmx = Runtime.getRuntime().maxMemory();
        long javacppMaxPhys = Pointer.maxPhysicalBytes();

        List<String[]> table = new ArrayList<>();
        table.add(new String[]{"Version", inferVersion()});
        table.add(new String[]{"Name", name});
        table.add(new String[]{"Description", description});
        table.add(new String[]{"Operating System",
                os.getManufacturer() + " " +
                        os.getFamily() + " " +
                        os.getVersion().getVersion()});
        table.add(new String[]{"Devices", devices().get(0)});
        table.add(new String[]{"CPU Cores", cpuCores});
        table.add(new String[]{"CPU", procName});
        table.add(new String[]{"System Memory", formatBytes(totalMem)});
        table.add(new String[]{"Memory Config - XMX", formatBytes(xmx)});
        table.add(new String[]{"Memory Config - JavaCPP MaxPhysicalBytes", formatBytes(javacppMaxPhys)});
        table.add(new String[]{"Backend", backend});
        table.add(new String[]{"ND4J DataType", Nd4j.dataType().toString()});
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
        table.add(new String[]{"Total Params", "" + numParams});
        table.add(new String[]{"Total Layers", Integer.toString(numLayers)});
        table.add(new String[]{"Batch size", Integer.toString(batchSize)});

        //Fit
        double fitMs = avgFitTime();
        double fitStd = stdFitTime();
        double fitBatchesPerSec = 1000.0 / fitMs;
        double fitExamplesPerSec = fitBatchesPerSec * batchSize;
        table.add(new String[]{"Fit time (ms): ", df.format(fitMs)});
        table.add(new String[]{"Fit time stdev (ms): ", df.format(fitStd)});
        table.add(new String[]{"Fit batches/sec: ", df.format(fitBatchesPerSec)});
        table.add(new String[]{"Fit examples/sec: ", df.format(fitExamplesPerSec)});

        //Forward
        double fwdMs = avgForwardTime();
        double fwdStd = stdForwardTime();
        double fwdBatchesPerSec = 1000.0 / fwdMs;
        double fwdExamplesPerSec = fwdBatchesPerSec * batchSize;
        table.add(new String[]{"Forward time (ms): ", df.format(fwdMs)});
        table.add(new String[]{"Forward time stdev (ms): ", df.format(fwdStd)});
        table.add(new String[]{"Forward batches/sec: ", df.format(fwdBatchesPerSec)});
        table.add(new String[]{"Forward examples/sec: ", df.format(fwdExamplesPerSec)});

        //Backward
        double bwdMs = avgBackpropTime();
        double bwdStd = stdBackpropTime();
        double bwdBatchesPerSec = 1000.0 / bwdMs;
        double bwdExamplesPerSec = bwdBatchesPerSec * batchSize;
        table.add(new String[]{"Backward time (ms): ", df.format(bwdMs)});
        table.add(new String[]{"Backward time stdev (ms): ", df.format(bwdStd)});
        table.add(new String[]{"Backward batches/sec: ", df.format(bwdBatchesPerSec)});
        table.add(new String[]{"Backward examples/sec: ", df.format(bwdExamplesPerSec)});

        StringBuilder sb = new StringBuilder();

        for (final Object[] row : table) {
            sb.append(String.format("%-42s %-45s\n", row));
        }

        return sb.toString();
    }

    private static String formatBytes(long bytes){
        return bytes + " - " + StringUtils.TraditionalBinaryPrefix.long2String(bytes, null, 2);
    }
}
