package org.tensorflow.model.examples.regression.linear;

import org.tensorflow.Graph;
import org.tensorflow.Result;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Pow;
import org.tensorflow.types.TFloat32;

import java.util.Random;


public class LinearRegressionExample {
    /**
     * This value is used to fill the Y placeholder in prediction.
     */
    public static final float LEARNING_RATE = 0.1f;
    public static final String WEIGHT_VARIABLE_NAME = "weight";
    public static final String BIAS_VARIABLE_NAME = "bias";
    /**
     * Amount of data points.
     */
    private static final int N = 15;

    public static void main(String[] args) {
        float[] xValues = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f};
        float[] yValues = new float[N];

        Random rnd = new Random(42);

        for (int i = 0; i < yValues.length; i++) {
            yValues[i] = (float) (10 * xValues[i] + 2 + 0.1 * (rnd.nextDouble() - 0.5));
        }

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Define placeholders
            Placeholder<TFloat32> xData = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.scalar()));
            Placeholder<TFloat32> yData = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.scalar()));

            // Define variables
            Variable<TFloat32> weight = tf.withName(WEIGHT_VARIABLE_NAME).variable(tf.constant(1f));
            Variable<TFloat32> bias = tf.withName(BIAS_VARIABLE_NAME).variable(tf.constant(1f));

            // Define the model function weight*x + bias
            Mul<TFloat32> mul = tf.math.mul(xData, weight);
            Add<TFloat32> yPredicted = tf.math.add(mul, bias);

            // Define loss function MSE
            Pow<TFloat32> sum = tf.math.pow(tf.math.sub(yPredicted, yData), tf.constant(2f));
            Div<TFloat32> mse = tf.math.div(sum, tf.constant(2f * N));

            // Back-propagate gradients to variables for training
            Optimizer optimizer = new GradientDescent(graph, LEARNING_RATE);
            Op minimize = optimizer.minimize(mse);

            try (Session session = new Session(graph)) {

                // Train the model on data
                for (int i = 0; i < xValues.length; i++) {
                    float y = yValues[i];
                    float x = xValues[i];

                    try (TFloat32 xTensor = TFloat32.scalarOf(x);
                         TFloat32 yTensor = TFloat32.scalarOf(y)) {

                        session.runner()
                                .addTarget(minimize)
                                .feed(xData.asOutput(), xTensor)
                                .feed(yData.asOutput(), yTensor)
                                .run();

                        System.out.println("Training phase");
                        System.out.println("x is " + x + " y is " + y);
                    }
                }

                // Extract linear regression model weight and bias values
                Result tensorList = session.runner()
                        .fetch(WEIGHT_VARIABLE_NAME)
                        .fetch(BIAS_VARIABLE_NAME)
                        .run();

                try (TFloat32 weightValue = (TFloat32) tensorList.get(0);
                     TFloat32 biasValue = (TFloat32) tensorList.get(1)) {

                    System.out.println("Weight is " + weightValue.getFloat());
                    System.out.println("Bias is " + biasValue.getFloat());
                }

                // Let's predict y for x = 10f
                float x = 10f;
                float predictedY = 0f;

                try (TFloat32 xTensor = TFloat32.scalarOf(x);
                     TFloat32 yTensor = TFloat32.scalarOf(predictedY);
                     TFloat32 yPredictedTensor = (TFloat32) session.runner()
                             .feed(xData.asOutput(), xTensor)
                             .feed(yData.asOutput(), yTensor)
                             .fetch(yPredicted)
                             .run().get(0)) {

                    predictedY = yPredictedTensor.getFloat();

                    System.out.println("Predicted value: " + predictedY);
                }
            }
        }
    }
}
