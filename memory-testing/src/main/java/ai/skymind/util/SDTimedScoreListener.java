package ai.skymind.util;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.dataset.api.MultiDataSet;

@Slf4j
public class SDTimedScoreListener extends BaseListener {

    private final long printMs;
    private long last;

    public SDTimedScoreListener(long printMs){
        this.printMs = printMs;
    }

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        if(last == 0){
            last = System.currentTimeMillis();
            return;
        }

        if(last + printMs < System.currentTimeMillis()){
            double score = loss.totalLoss();
            log.info("Score at epoch {} iteration {} is {}", at.epoch(), at.iteration(), score);
            last = System.currentTimeMillis();
        }
    }
}
