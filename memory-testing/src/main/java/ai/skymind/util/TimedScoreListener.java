package ai.skymind.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

@Slf4j
public class TimedScoreListener extends BaseTrainingListener {

    private final long printMs;
    private long last;

    public TimedScoreListener(long printMs){
        this.printMs = printMs;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(last == 0){
            last = System.currentTimeMillis();
            return;
        }

        if(last + printMs < System.currentTimeMillis()){
            double score = model.score();
            log.info("Score at iteration {} is {}", iteration, score);
            last = System.currentTimeMillis();
        }
    }

}
