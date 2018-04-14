package org.deeplearning4j.memory;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.sets.IntegerListOptionHandler;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.OptionDef;
import org.kohsuke.args4j.spi.OneArgumentOptionHandler;
import org.kohsuke.args4j.spi.Setter;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Slf4j
public class BenchmarkCnnMemory extends BaseMemoryBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.VGG16;
    @Option(name="--numLabels",usage="Train batch size.",aliases = "-labels")
    public static int numLabels = 1000;
    @Option(name="--batchSizes",usage="Train batch size.",aliases = "-batch", required = true, handler = IntegerListOptionHandler.class)
    public static List<Integer> batchSizes = new ArrayList<>();
//    public static List<Integer> batchSizes = Arrays.asList(1,2,4,8,16);
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 5000;
    @Option(name="--memoryTest", usage = "Type of memory test")
    public static MemoryTest memoryTest = MemoryTest.TRAINING;
    @Option(name="--workspaceMode", usage = "Workspace mode to use")
    public static WorkspaceMode workspaceMode = WorkspaceMode.SEPARATE;

    private String datasetName  = "SIMULATEDCNN";
    private int seed = 42;

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

        log.info("Building models for "+modelType+"....");
        Map<ModelType, TestableModel> map = ModelSelector.select(modelType, null, numLabels, seed, 1);
        if(map.size() != 1){
            throw new IllegalStateException();
        }

        ModelType mt = map.keySet().iterator().next();
        TestableModel net = map.get(mt);

        int[][] inputShape = net.metaData().getInputShape();
        String description = datasetName + " " + batchSizes + "x" + inputShape[0][0] + "x" + inputShape[0][1] + "x" + inputShape[0][2];

        log.info("Preparing memory benchmark: {}", description);
        String name = mt.toString();
        benchmark(name, description, mt, net, memoryTest, batchSizes, workspaceMode);

        System.exit(0);
    }

    public static void main(String[] args) throws Exception {

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(false);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        new BenchmarkCnnMemory().run(args);
    }


    public static class IntArrayOptionHandler extends OneArgumentOptionHandler<int[]> {

        public IntArrayOptionHandler(CmdLineParser parser, OptionDef option, Setter<? super int[]> setter) {
            super(parser, option, setter);
        }

        @Override
        protected int[] parse(String s) throws NumberFormatException, CmdLineException {
            s = s.replaceAll(" |\\[|\\]","");
            String[] split = s.split(",");
            int[] out = new int[split.length];
            for( int i=0; i<split.length; i++ ){
                out[i] = Integer.parseInt(split[i]);
            }
            return out;
        }
    }
}
