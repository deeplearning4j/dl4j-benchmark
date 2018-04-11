package org.deeplearning4j.listeners;

/**
 * Created by justin on 3/25/17.
 */

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A listener that collects statistics important for evaluating the discrimination of
 * a training process. This includes the area under a ROC curve developed from key points
 * during a training process. Statistics collected here are meant to be compared offline
 * for assessment.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TrainingDiscriminationListener implements TrainingListener {
    private final int frequency;
    private final boolean reportRocArea;
    private final boolean reportScore;

    private int xCount;
    private List<Double> curveX;
    private List<Double> curveY;

    public TrainingDiscriminationListener() {
        this(1, false, true);
    }

    public TrainingDiscriminationListener(int frequency) {
        this(frequency, false, true);
    }

    public TrainingDiscriminationListener(int frequency, boolean reportScore, boolean reportRocArea) {
        this.frequency = frequency;
        this.reportRocArea = reportRocArea;
        this.reportScore = reportScore;
        this.curveX = new ArrayList<>();
        this.curveY = new ArrayList<>();
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(iteration % frequency == 0) {
            ++this.xCount;

            // each score is treated as a point in a polygon
            // for the purpose of calculating area
            curveX.add((double) xCount);
            curveY.add(model.score());

            if(reportRocArea) {
                // add bottom right coordinates
                curveX.add((double) xCount);
                curveY.add(0.0);
                // merge curve with two y=0 points to form proper polygon
                double[] pointsX = ArrayUtils.addAll(new double[]{0.0}, ArrayUtils.toPrimitive(curveX.toArray(new Double[0])));
                double[] pointsY = ArrayUtils.addAll(new double[]{0.0}, ArrayUtils.toPrimitive(curveY.toArray(new Double[0])));
                double area = calculateArea(pointsX, pointsY, pointsX.length);
                log.info("Score curve area at iteration " + iteration + " is " + area);
            }
            if(reportScore) log.info("Score at iteration " + iteration + " is " + model.score());
        }
    }

    private double calculateArea(double[] pointsX, double[] pointsY, int nPoints) {
        double sum = 0;
        for (int i = 0; i < nPoints ; i++)
        {
            sum = sum + pointsX[i]*pointsY[(i+1)%nPoints] - pointsY[i]*pointsX[(i+1)%nPoints];
        }

        return Math.abs(sum / 2);
    }

    public void onEpochStart(Model var1) {
        // no op
    }

    public void onEpochEnd(Model var1) {
        // no op
    }

    public void onForwardPass(Model var1, List<INDArray> var2) {
        // no op
    }

    public void onForwardPass(Model var1, Map<String, INDArray> var2) {
        // no op
    }

    public void onGradientCalculation(Model var1) {
        // no op
    }

    public void onBackwardPass(Model var1) {
        // no op
    }
}
