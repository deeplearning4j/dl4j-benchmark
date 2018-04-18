package org.deeplearning4j.sets;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.OptionDef;
import org.kohsuke.args4j.spi.Messages;
import org.kohsuke.args4j.spi.OptionHandler;
import org.kohsuke.args4j.spi.Parameters;
import org.kohsuke.args4j.spi.Setter;

public class IntegerListOptionHandler extends OptionHandler<Integer>  {

    public IntegerListOptionHandler(CmdLineParser parser, OptionDef option, Setter<Integer> setter) {
        super(parser, option, setter);
    }

    public String getDefaultMetaVariable() {
        return Messages.DEFAULT_META_STRING_ARRAY_OPTION_HANDLER.format(new Object[0]);
    }

    public int parseArguments(Parameters params) throws CmdLineException {
        int counter;
        for(counter = 0; counter < params.size(); ++counter) {
            String param = params.getParameter(counter);
            if (param.startsWith("-")) {
                break;
            }

            String[] arr$ = param.split(" ");
            int len$ = arr$.length;

            for(int i$ = 0; i$ < len$; ++i$) {
                String p = arr$[i$];
                this.setter.addValue(Integer.parseInt(p));
            }
        }

        return counter;
    }
}
