package org.tensorAction;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.MaxPool;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;

import javax.imageio.IIOException;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class tensorTrainer {
    private static final int PIXEL_DEPTH = 255;
    private static final int NUM_CHANNELS = 3;
    private static final int IMAGE_SIZE = 32;
    private static final long SEED = 123456789L;
    private static final String PADDING_TYPE = "SAME";
    static File[] classDirs;
    static List<TFloat32> imageTensors;
    static List<TFloat32> labelTensors;

    public static Graph build() {
        Graph graph = new Graph();
        Ops tf = Ops.create(graph);

        // Inputs
        Placeholder<TUint8> input = tf.withName("input").placeholder(TUint8.class, Placeholder.shape(Shape.of(-1, IMAGE_SIZE, IMAGE_SIZE)));
        Reshape<TUint8> input_reshaped = tf.reshape(input, tf.array(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS));
        Placeholder<TUint8> labels = tf.withName("target").placeholder(TUint8.class);

        // Scaling the features
        Constant<TFloat32> centeringFactor = tf.constant(PIXEL_DEPTH / 2.0f);
        Constant<TFloat32> scalingFactor = tf.constant((float) PIXEL_DEPTH);
        Operand<TFloat32> scaledInput = tf.math.div(tf.math.sub(tf.dtypes.cast(input_reshaped, TFloat32.class), centeringFactor), scalingFactor);

        // First conv layer
        Variable<TFloat32> conv1Weights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(5, 5, NUM_CHANNELS, 32), TFloat32.class, TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Conv2d<TFloat32> conv1 = tf.nn.conv2d(scaledInput, conv1Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> conv1Biases = tf.variable(tf.fill(tf.array(32), tf.constant(0.0f)));
        Relu<TFloat32> relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases));

        // First pooling layer
        MaxPool<TFloat32> pool1 = tf.nn.maxPool(relu1, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1), PADDING_TYPE);

        // Second conv layer
        Variable<TFloat32> conv2Weights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(5, 5, 32, 64), TFloat32.class, TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Conv2d<TFloat32> conv2 = tf.nn.conv2d(pool1, conv2Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> conv2Biases = tf.variable(tf.fill(tf.array(64), tf.constant(0.1f)));
        Relu<TFloat32> relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases));

        // Second pooling layer
        MaxPool<TFloat32> pool2 = tf.nn.maxPool(relu2, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1), PADDING_TYPE);

        // Flatten inputs
        Reshape<TFloat32> flatten = tf.reshape(pool2, tf.concat(Arrays.asList(tf.slice(tf.shape(pool2), tf.array(0), tf.array(1)), tf.array(-1)), tf.constant(0)));

        // Fully connected layer
        Variable<TFloat32> fc1Weights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(IMAGE_SIZE * IMAGE_SIZE * 4, 512), TFloat32.class, TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc1Biases = tf.variable(tf.fill(tf.array(512), tf.constant(0.1f)));
        Relu<TFloat32> relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases));

        // Softmax layer
        Variable<TFloat32> fc2Weights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(512, classDirs.length), TFloat32.class, TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc2Biases = tf.variable(tf.fill(tf.array(classDirs.length), tf.constant(0.1f)));

        Add<TFloat32> logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases);

        // Predicted outputs
        tf.withName("output").nn.softmax(logits);

        // Loss function & regularization
        OneHot<TFloat32> oneHot = tf.oneHot(labels, tf.constant(classDirs.length), tf.constant(1.0f), tf.constant(0.0f));
        SoftmaxCrossEntropyWithLogits<TFloat32> batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, oneHot);
        Mean<TFloat32> labelLoss = tf.math.mean(batchLoss.loss(), tf.constant(0));
        Add<TFloat32> regularizes = tf.math.add(tf.nn.l2Loss(fc1Weights), tf.math.add(tf.nn.l2Loss(fc1Biases), tf.math.add(tf.nn.l2Loss(fc2Weights), tf.nn.l2Loss(fc2Biases))));
        Add<TFloat32> loss = tf.withName("training_loss").math.add(labelLoss, tf.math.mul(regularizes, tf.constant(8e-4f)));

        // Optimizer
        Optimizer optimizer = new Adam(graph, 0.001f, 0.9f, 0.999f, 8e-4f);

        System.out.println("Optimizer = " + optimizer);
        optimizer.minimize(loss, "train");

        return graph;
    }

    public static void main(String[] args) throws IOException {
        String dataDir = "/Users/gregor/Desktop/Tensorflow_ObjD/flower_photos";
        access(dataDir);
    }

    public static void train(List<TFloat32> imageTensors, List<TFloat32> labelTensors) {
        int batchSize = 32;
        int numBatches = (int) Math.ceil(imageTensors.size() / (double) batchSize);

        try (Graph graph = build()) {
            try (Session session = new Session(graph)) {
                for (int epoch = 0; epoch < 100; epoch++) {
                    for (int batch = 0; batch < numBatches; batch++) {
                        long start = (long) batch * batchSize;
                        long end = Math.min(start + batchSize, imageTensors.size());

                        List<TFloat32> batchImages = imageTensors.subList((int) start, (int) end);
                        List<TFloat32> batchLabels = labelTensors.subList((int) start, (int) end);

                        try (TUint8 batchImagesTensor = TUint8.tensorOf(NdArrays.ofBytes(Shape.of(batchImages.size(), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)));
                             TUint8 batchLabelsTensor = TUint8.tensorOf(NdArrays.ofBytes(Shape.of(batchLabels.size())))) {

                            for (int i = 0; i < batchImages.size(); i++) {
                                batchImagesTensor.slice(Indices.at(i)).copyFrom(batchImages.get(i).asRawTensor().data());
                                batchLabelsTensor.slice(Indices.at(i)).copyFrom(batchLabels.get(i).asRawTensor().data());
                            }

                            TFloat32 loss = (TFloat32) session.runner()
                                    .feed("target", batchLabelsTensor)
                                    .feed("input", batchImagesTensor)
                                    .addTarget("train")
                                    .fetch("training_loss")
                                    .run().get(0);
                            System.out.println("Epoch " + epoch + ", Batch " + batch + " Loss: " + loss.getFloat());
                            test2(session, imageTensors, labelTensors);
                        }
                    }
                }

                test(session, imageTensors, labelTensors);

                try {
                    saveModel(graph, session, "/Users/gregor/Desktop");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public static void test(Session session, List<TFloat32> imageTensors, List<TFloat32> labelTensors) {
        int correctCount = 0;
        int[][] confusionMatrix = new int[classDirs.length][classDirs.length];

        // Iterate over each test data sample
        for (int i = 0; i < imageTensors.size(); i++) {
            try (TUint8 transformedInput = TUint8.tensorOf(imageTensors.get(i).asRawTensor().shape())) {
                transformedInput.copyFrom(imageTensors.get(i).asRawTensor().data());

                // Perform prediction
                TFloat32 outputTensor = (TFloat32) session.runner()
                        .feed("input", transformedInput)
                        .fetch("output")
                        .run()
                        .get(0);

                // Convert trueLabelTensor to an integer
                int trueLabel = argmax(labelTensors.get(i));

                // Get predicted label
                int predLabel = argmax2(outputTensor.slice(Indices.at(0), Indices.all()));

                // Compare prediction with true label
                if (predLabel == trueLabel) {
                    correctCount++;

                    // Update confusion matrix
                    confusionMatrix[trueLabel][predLabel]++;
                }
            }
        }

        System.out.println("Final accuracy = " + (((float) correctCount) / imageTensors.size()) * 100 + "%");
        StringBuilder sb = getStringBuilder(confusionMatrix);
        System.out.println(sb);
    }
    public static void test2(Session session, List<TFloat32> imageTensors, List<TFloat32> labelTensors) {
        int correctCount = 0;

        // Iterate over each test data sample
        for (int i = 0; i < imageTensors.size(); i++) {
            try (TUint8 transformedInput = TUint8.tensorOf(imageTensors.get(i).asRawTensor().shape())) {
                transformedInput.copyFrom(imageTensors.get(i).asRawTensor().data());

                // Perform prediction
                TFloat32 outputTensor = (TFloat32) session.runner()
                        .feed("input", transformedInput)
                        .fetch("output")
                        .run()
                        .get(0);

                // Convert trueLabelTensor to an integer
                int trueLabel = argmax(labelTensors.get(i));

                // Get predicted label
                int predLabel = argmax2(outputTensor.slice(Indices.at(0), Indices.all()));

                // Compare prediction with true label
                if (predLabel == trueLabel) {
                    correctCount++;
                }
            }
        }

        System.out.print("Final accuracy = " + (((float) correctCount) / imageTensors.size()) * 100 + "% - ");
    }

    private static StringBuilder getStringBuilder(int[][] confusionMatrix) {
        StringBuilder sb = new StringBuilder();
        sb.append("Label");
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%1$5s", "" + i));
        }
        sb.append("\n");

        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%1$5s", "" + i));
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                sb.append(String.format("%1$5s", "" + confusionMatrix[i][j]));
            }
            sb.append("\n");
        }
        return sb;
    }

    public static int argmax(FloatNdArray probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.shape().get(0); i++) {
            float curVal = probabilities.getFloat(i);
            if (curVal > maxVal) {
                maxVal = curVal;
                idx = i;
            }
        }
        return idx;
    }

    public static int argmax2(FloatNdArray probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.shape().get(0); i++) {
            float curVal = probabilities.getFloat(i);

            if (curVal > maxVal && curVal < 1 && curVal > 0) {
                maxVal = curVal;
                idx = i;
            }
        }
        return idx;
    }

    public static void loadDataset(String dataDir, List<TFloat32> imageTensors, List<TFloat32> labelTensors) throws IOException {
        classDirs = new File(dataDir).listFiles(File::isDirectory);
        if (classDirs != null) {
            Map<String, Integer> classLabelMap = new HashMap<>();
            for (int i = 0; i < classDirs.length; i++) {
                classLabelMap.put(classDirs[i].getName(), i);
            }

            for (File classDir : classDirs) {
                String className = classDir.getName();
                int classLabel = classLabelMap.get(className);
                File[] imageFiles = classDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg"));
                if (imageFiles != null) {
                    for (File imageFile : imageFiles) {
                        try {
                            BufferedImage img = ImageIO.read(imageFile);
                            if (img != null) {
                                TFloat32 imageTensor = preprocessImage(img);
                                TFloat32 labelTensor = preprocessLabel(classLabel, classDirs.length);
                                imageTensors.add(imageTensor);
                                labelTensors.add(labelTensor);
                            }
                        } catch (IIOException e) {
                            System.out.println("Error reading image file: " + imageFile.getName() + " - " + e.getMessage());
                        }
                    }
                }
            }
        }
    }

    private static TFloat32 preprocessLabel(int classLabel, int numClasses) {
        float[] labelArray = new float[numClasses];
        labelArray[classLabel] = 1.0f;
        return TFloat32.tensorOf(StdArrays.ndCopyOf(labelArray));
    }

    private static TFloat32 preprocessImage(BufferedImage img) {
        float[] imgData = new float[IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS];
        int[] rgbArray = img.getRGB(0, 0, IMAGE_SIZE, IMAGE_SIZE, null, 0, IMAGE_SIZE);
        for (int i = 0; i < rgbArray.length; i++) {
            int pixel = rgbArray[i];
            imgData[i * 3] = ((pixel >> 16) & 0xFF) / 255.0f;
            imgData[i * 3 + 1] = ((pixel >> 8) & 0xFF) / 255.0f;
            imgData[i * 3 + 2] = (pixel & 0xFF) / 255.0f;
        }
        return TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][][]{new float[][]{imgData}}));
    }


    private static void saveModel(Graph graph, Session session, String exportDir) throws IOException {
        session.save("/Users/gregor/Desktop/model");
        Files.write(Paths.get(exportDir, "model.pb"), graph.toGraphDef().toByteArray());
        System.out.println("Model saved to " + exportDir + "/model.pb");
    }

    public static void access(String folder) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        imageTensors = new ArrayList<>();
        labelTensors = new ArrayList<>();

        // Load and preprocess the dataset
        loadDataset(folder, imageTensors, labelTensors);
        System.out.println("Dataset loaded. Number of photos: " + imageTensors.size());

        train(imageTensors, labelTensors);
    }
}