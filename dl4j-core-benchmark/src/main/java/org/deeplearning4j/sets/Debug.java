package org.deeplearning4j.sets;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.util.Arrays;

public class Debug {

    @Option(name="--batchSizes",usage="Train batch size.",aliases = "-batch", required = false) //, handler = IntArrayOptionHandler.class)
    public static int[] batchSizes = {1, 2, 4, 8, 16};

    public static void main(String... args) throws Exception {
        new Debug().run(args);
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


        System.out.println(Arrays.toString(batchSizes));
    }

}
