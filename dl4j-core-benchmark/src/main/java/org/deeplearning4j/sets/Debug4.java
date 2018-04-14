package org.deeplearning4j.sets;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.spi.StringArrayOptionHandler;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Debug4 {

    @Option(name="--batchSizes",usage="Train batch size.",aliases = "-batch", required = true, handler = StringArrayOptionHandler.class) //, handler = IntArrayOptionHandler.class)
    public static List<String> batchSizes = new ArrayList<>();

    public static void main(String... args) throws Exception {
        new Debug4().run(args);
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


        System.out.println(batchSizes);
    }

}
