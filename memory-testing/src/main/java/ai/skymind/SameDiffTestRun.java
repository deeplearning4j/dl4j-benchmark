package ai.skymind;

import ai.skymind.util.SDTimedScoreListener;
import ai.skymind.util.Utils;
import lombok.extern.slf4j.Slf4j;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class SameDiffTestRun {

    @Option(name = "--modelClass", usage = "Model class", required = true)
    public static String modelClass;
    @Option(name = "--dataClass", usage = "Data pipeline class", required = true)
    public static String dataClass;
    @Option(name = "--runtimeSec", usage = "Maximum runtime (seconds)")
    public static int runtimeSec = 3600;    //1 hour
    @Option(name = "--periodicGC", usage = "Periodic GC frequency (<= 0 is disabled - default)")
    public static int periodicGC = 0;
    @Option(name = "--useHelpers", usage = "Whether to use MKL-DNN/cuDNN or not")
    public static boolean useHelpers = false;

    public static void main(String[] args) throws Exception {
        new SameDiffTestRun().run(args);
    }

    public void run(String[] args) throws Exception {
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            System.exit(1);
        }

        log.info("Model class: {}", modelClass);
        log.info("Data class: {}", dataClass);
        log.info("Runtime: {} seconds", runtimeSec);
        log.info("Periodic GC: {}", (periodicGC <= 0 ? "disabled" : periodicGC + " ms"));
        log.info("Use helpers: {}", useHelpers);

        if(!useHelpers){
            Utils.disableMKLDNN();
        }

        Utils.logMemoryConfig();
        AtomicLong[] bytes = Utils.startMemoryLoggingThread(30000);

        SameDiffModel m = (SameDiffModel) Class.forName(modelClass).newInstance();
        Pipeline p = (Pipeline) Class.forName(dataClass).newInstance();

        SameDiff model = m.getModel();
        model.setListeners(new SDTimedScoreListener(60000));

        if(periodicGC > 0) {
            Nd4j.getMemoryManager().togglePeriodicGc(true);
            Nd4j.getMemoryManager().setAutoGcWindow(periodicGC);
        }

        long start = System.currentTimeMillis();
        long end = start + runtimeSec * 1000L;
        switch (p.type()){
            case DATASET_ITERATOR:
                DataSetIterator iter = p.getIterator();
                while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                    model.fit(iter, 1);
                }
                break;
            case MDS_ITERATOR:
                MultiDataSetIterator mdsIter = p.getMdsIterator();
                while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                    model.fit(mdsIter, 1);
                }
                break;
            case INDARRAYS:
                int iterCount = 0;
                long lastReport = System.currentTimeMillis();
                List<String> inputs = model.inputs();
                List<String> outputs = model.outputs();
                Map<String,INDArray> phMap = new HashMap<>();
                while(System.currentTimeMillis() < end){
                    INDArray[] next = p.getFeatures();
                    for(int i=0; i<next.length; i++ ){
                        phMap.put(inputs.get(i), next[i]);
                    }
                    model.exec(phMap, outputs);
                    if(++iterCount % 1000 == 0 && System.currentTimeMillis() > lastReport + 60000L ){
                        log.info("Num iters: {}", iterCount);
                        lastReport = System.currentTimeMillis();
                    }
                }
                break;
            default:
                throw new RuntimeException("Unknown data type: " + p.type());
        }

        log.info("Completed. Max observed bytes = {}, max physical bytes = {}", bytes[0].get(), bytes[1].get());
    }

}
