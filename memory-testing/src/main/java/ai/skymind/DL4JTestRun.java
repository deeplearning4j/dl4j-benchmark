package ai.skymind;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

@Slf4j
public class DL4JTestRun {

    @Option(name = "--modelClass", usage = "Model class", required = true)
    public static String modelClass;
    @Option(name = "--dataClass", usage = "Data pipeline class", required = true)
    public static String dataClass;
    @Option(name = "--runtimeSec", usage = "Maximum runtime (seconds)")
    public static int runtimeSec = 3600;    //1 hour

    public static void main(String[] args) throws Exception {
        new DL4JTestRun().run(args);
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

        Utils.logMemoryConfig();
        Utils.startMemoryLoggingThread(1000);

        BenchmarkModel m = (BenchmarkModel) Class.forName(modelClass).newInstance();
        Pipeline p = (Pipeline) Class.forName(dataClass).newInstance();

        Model model = m.getModel();
        boolean mln = model instanceof MultiLayerNetwork;
        model.setListeners(new ScoreIterationListener(1));

        long start = System.currentTimeMillis();
        long end = start + runtimeSec * 1000L;
        switch (p.type()){
            case DATASET_ITERATOR:
                DataSetIterator iter = p.getIterator();
                while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                    if(mln){
                        ((MultiLayerNetwork)model).fit(iter);
                    } else {
                        ((ComputationGraph)model).fit(iter);
                    }
                }
                break;
            case MDS_ITERATOR:
                MultiDataSetIterator mdsIter = p.getMdsIterator();
                while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                    if(mln){
                        ((MultiLayerNetwork)model).fit(mdsIter);
                    } else {
                        ((ComputationGraph)model).fit(mdsIter);
                    }
                }
                break;
            case INDARRAYS:
                while(System.currentTimeMillis() < end){
                    INDArray[] next = p.getFeatures();
                    if(mln){
                        ((MultiLayerNetwork)model).output(next[0]);
                    } else {
                        ((ComputationGraph)model).output(next);
                    }
                }
                break;
            default:
                throw new RuntimeException("Unknown data type: " + p.type());
        }
    }

}
