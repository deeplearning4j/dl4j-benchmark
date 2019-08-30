package ai.skymind;

import ai.skymind.util.SDTimedScoreListener;
import ai.skymind.util.TimedScoreListener;
import ai.skymind.util.Utils;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncShieldDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncShieldMultiDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.StringArrayOptionHandler;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class TestRun {

    @Option(name = "--modelClass", usage = "Model class(es)", handler = StringArrayOptionHandler.class, required = true)
    public static List<String> modelClasses;
    @Option(name = "--dataClass", usage = "Data pipeline class(es)", handler = StringArrayOptionHandler.class, required = true)
    public static List<String> dataClasses;
    @Option(name = "--runtimeSec", usage = "Maximum runtime (seconds) for each model")
    public static int runtimeSec = 3600;    //1 hour
    @Option(name = "--periodicGC", usage = "Periodic GC frequency (<= 0 is disabled - default)")
    public static int periodicGC = 0;
    @Option(name = "--useHelpers", usage = "Whether to use MKL-DNN/cuDNN or not")
    public static boolean useHelpers = false;
    @Option(name = "--maxIters", usage = "Maximum number of iterations")
    public static int maxIters = Integer.MAX_VALUE;
    @Option(name = "--asyncShield", usage = "Whether an async shield should be added")
    public static boolean asyncShield = false;
    @Option(name = "--debugMode", usage = "Debug mode: should we log every iteration?")
    public static boolean debugMode = false;

    public static void main(String[] args) throws Exception {
        new TestRun().run(args);
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

        INDArray array = Nd4j.create(1);
        array.addi(3);
        log.info("Array confirmation: {}", array);
        log.info("Model classes: {}", modelClasses);
        log.info("Data classes: {}", dataClasses);
        log.info("Runtime: {} seconds", runtimeSec);
        log.info("Periodic GC: {}", (periodicGC <= 0 ? "disabled" : periodicGC + " ms"));
        log.info("Use helpers: {}", useHelpers);
        log.info("Max iters: {}", maxIters);
        log.info("Async shield: {}", asyncShield);
        log.info("Debug mode: {}", debugMode);


        Preconditions.checkState(modelClasses.size() == dataClasses.size(), "Number of model classes (%s) must match " +
                "number of data classes (%s)", modelClasses.size(), dataClasses.size());

        if (!useHelpers) {
            Utils.disableMKLDNN();
        }

        Utils.logMemoryConfig();
        AtomicLong[] bytes = Utils.startMemoryLoggingThread(30000);

        if(periodicGC > 0) {
            Nd4j.getMemoryManager().togglePeriodicGc(true);
            Nd4j.getMemoryManager().setAutoGcWindow(periodicGC);
        }

        for( int i=0; i<modelClasses.size(); i++ ){
            log.info("========== Starting execution: model {} of {} ==========", (i+1), modelClasses.size());
            log.info("Model: {}", modelClasses.get(i));
            log.info("Data: {}", dataClasses.get(i));

            Class<?> c = Class.forName(modelClasses.get(i));
            Pipeline p = (Pipeline) Class.forName(dataClasses.get(i)).newInstance();

            if(BenchmarkModel.class.isAssignableFrom(c)){
                //DL4J Model
                Model model = ((BenchmarkModel)c.newInstance()).getModel();
                log.info("Staring test for DL4J model - {}", model.getClass().getSimpleName());

                if(debugMode){
                    model.setListeners(new ScoreIterationListener(1));
                } else {
                    model.setListeners(new TimedScoreListener(60000));
                }

                boolean mln = model instanceof MultiLayerNetwork;


                if(!useHelpers){
                    Utils.removeHelpers(model);
                }

                long start = System.currentTimeMillis();
                long end = start + runtimeSec * 1000L;
                switch (p.type()){
                    case DATASET_ITERATOR:
                        DataSetIterator iter = p.getIterator();
                        if(asyncShield)
                            iter = new AsyncShieldDataSetIterator(iter);

                        while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                            if(mln){
                                ((MultiLayerNetwork)model).fit(iter);
                                if(((MultiLayerNetwork) model).getIterationCount() >= maxIters)
                                    break;  //TODO break before epoch is done if iter limit is hit
                            } else {
                                ((ComputationGraph)model).fit(iter);
                                if(((ComputationGraph) model).getIterationCount() >= maxIters)
                                    break;  //TODO break before epoch is done if iter limit is hit
                            }
                        }
                        break;
                    case MDS_ITERATOR:
                        MultiDataSetIterator mdsIter = p.getMdsIterator();
                        if(asyncShield)
                            mdsIter = new AsyncShieldMultiDataSetIterator(mdsIter);

                        while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                            if(mln){
                                ((MultiLayerNetwork)model).fit(mdsIter);
                                if(((MultiLayerNetwork) model).getIterationCount() >= maxIters)
                                    break;  //TODO break before epoch is done if iter limit is hit
                            } else {
                                ((ComputationGraph)model).fit(mdsIter);
                                if(((ComputationGraph) model).getIterationCount() >= maxIters)
                                    break;  //TODO break before epoch is done if iter limit is hit
                            }
                        }
                        break;
                    case INDARRAYS:
                        int iterCount = 0;
                        long lastReport = System.currentTimeMillis();
                        while(System.currentTimeMillis() < end && iterCount < maxIters){
                            INDArray[] next = p.getFeatures();
                            if(mln){
                                ((MultiLayerNetwork)model).output(next[0]);
                            } else {
                                ((ComputationGraph)model).output(next);
                            }
                            if(++iterCount % 1000 == 0 && System.currentTimeMillis() > lastReport + 60000L ){
                                log.info("Num iters: {}", iterCount);
                                lastReport = System.currentTimeMillis();
                            }
                        }
                        break;
                    default:
                        throw new RuntimeException("Unknown data type: " + p.type());
                }

                log.info("Completed DL4J Model. Max observed bytes = {}, max physical bytes = {}", bytes[0].get(), bytes[1].get());


            } else {
                //SameDiff model
                log.info("Starting test for SameDiff model");

                SameDiff model = ((SameDiffModel)c.newInstance()).getModel();
                if(debugMode) {
                    model.setListeners(new ScoreListener(1));
                } else {
                    model.setListeners(new SDTimedScoreListener(60000));
                }

                long start = System.currentTimeMillis();
                long end = start + runtimeSec * 1000L;
                switch (p.type()){
                    case DATASET_ITERATOR:
                        DataSetIterator iter = p.getIterator();
                        if(asyncShield)
                            iter = new AsyncShieldDataSetIterator(iter);
                        while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                            model.fit(iter, 1);
                            if(model.getTrainingConfig().getIterationCount() > maxIters)
                                break;  //TODO break before epoch is done if iter limit is hit
                        }
                        break;
                    case MDS_ITERATOR:
                        MultiDataSetIterator mdsIter = p.getMdsIterator();
                        if(asyncShield)
                            mdsIter = new AsyncShieldMultiDataSetIterator(mdsIter);
                        while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                            model.fit(mdsIter, 1);
                            if(model.getTrainingConfig().getIterationCount() > maxIters)
                                break;  //TODO break before epoch is done if iter limit is hit
                        }
                        break;
                    case INDARRAYS:
                        int iterCount = 0;
                        long lastReport = System.currentTimeMillis();
                        List<String> inputs = model.inputs();
                        List<String> outputs = model.outputs();
                        Map<String, INDArray> phMap = new HashMap<>();
                        while(System.currentTimeMillis() < end && iterCount < maxIters){
                            INDArray[] next = p.getFeatures();
                            for(int j=0; j<next.length; j++ ){
                                phMap.put(inputs.get(j), next[j]);
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

                log.info("Completed SameDiff model. Max observed bytes = {}, max physical bytes = {}", bytes[0].get(), bytes[1].get());
            }

            log.info("========== Completed model {} of {} ==========", (i+1), modelClasses.size());
            System.gc();
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }
    }

}
