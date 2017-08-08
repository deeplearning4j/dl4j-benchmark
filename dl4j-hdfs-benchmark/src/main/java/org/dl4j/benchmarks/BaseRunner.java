package org.dl4j.benchmarks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;


/**
 *
 *
 * @author raver119@gmail.com
 */

public class BaseRunner {
    @Parameter(names = {"-p"}, description = "HDFS path where data is stored", required = true)
    private String pathHdfs = null;

    @Parameter(names = {"-k"}, description = "Local path where data is stored", required = false)
    private String pathLocal = null;

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



    }

    public static void main( String[] args ) throws Exception{
        new BaseRunner().run(args);
    }
}
