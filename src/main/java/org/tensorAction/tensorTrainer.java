package org.tensorAction;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.*;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class tensorTrainer {
    private static final int PIXEL_DEPTH = 255;
    private static final int NUM_CHANNELS = 3;
    private static final int IMAGE_SIZE = 40;
    private static final int NUM_LABELS = 10;
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
        Variable<TFloat32> fc2Weights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(512, NUM_LABELS), TFloat32.class, TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc2Biases = tf.variable(tf.fill(tf.array(NUM_LABELS), tf.constant(0.1f)));

        Add<TFloat32> logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases);

        // Predicted outputs
        tf.withName("output").nn.softmax(logits);

        // Loss function & regularization
        OneHot<TFloat32> oneHot = tf.oneHot(labels, tf.constant(10), tf.constant(1.0f), tf.constant(0.0f));
        SoftmaxCrossEntropyWithLogits<TFloat32> batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, oneHot);
        Mean<TFloat32> labelLoss = tf.math.mean(batchLoss.loss(), tf.constant(0));
        Add<TFloat32> regularizes = tf.math.add(tf.nn.l2Loss(fc1Weights), tf.math.add(tf.nn.l2Loss(fc1Biases), tf.math.add(tf.nn.l2Loss(fc2Weights), tf.nn.l2Loss(fc2Biases))));
        Add<TFloat32> loss = tf.withName("training_loss").math.add(labelLoss, tf.math.mul(regularizes, tf.constant(5e-4f)));

        // Optimizer
        Optimizer optimizer = new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f);

        System.out.println("Optimizer = " + optimizer);
        optimizer.minimize(loss, "train");

        return graph;
    }

    public static void main(String[] args) throws IOException {
        String dataDir = "/Users/gregor/Desktop/Tensorflow_ObjD/flower_photos";
        access(dataDir);
    }

    public static void train(ByteNdArray imageNdArray, ByteNdArray labelNdArray) {
        try (Graph graph = build()) {
            // Run the graph
            try (Session session = new Session(graph)) {
                // Train the model
                for (int i = 0; i < 1000; i++) {
                    try (TUint8 batchImages = TUint8.tensorOf(imageNdArray);
                         TUint8 batchLabels = TUint8.tensorOf(labelNdArray)) {
                        TFloat32 loss = (TFloat32) session.runner()
                                .feed("target", batchLabels)
                                .feed("input", batchImages)
                                .addTarget("train")
                                .fetch("training_loss")
                                .run().get(0);
                        System.out.println(loss.getFloat());
                    }
                }
                test(session, imageNdArray, labelNdArray);

                try {
                    saveModel(graph, session, "/Users/gregor/Desktop");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public static void test(Session session, ByteNdArray byteNdArray, ByteNdArray labeler) {
        int correctCount = 0;
        int[][] confusionMatrix = new int[10][10];

        for (int i = 0; i < 10; i++) {
            try (TUint8 transformedInput = TUint8.tensorOf(byteNdArray);
                 TFloat32 outputTensor = (TFloat32) session.runner()
                         .feed("input", transformedInput)
                         .fetch("output").run().get(0)) {

                for (int k = 0; k < labeler.shape().size(0); k++) {
                    byte trueLabel = labeler.getByte(k);
                    int predLabel;

                    predLabel = argmax(outputTensor.slice(Indices.at(k), Indices.all()));
                    if (predLabel == trueLabel) {
                        correctCount++;
                    }

                    confusionMatrix[trueLabel][predLabel]++;
                }
            }
        }

        System.out.println("Final accuracy = " + ((float) correctCount) / labeler.shape().size(0));
        StringBuilder sb = getStringBuilder(confusionMatrix);
        System.out.println(sb);
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
        for (int i = 0; i < probabilities.shape().size(0); i++) {
            float curVal = probabilities.getFloat(i);
            if (curVal > maxVal) {
                maxVal = curVal;
                idx = i;
            }
        }
        return idx;
    }

    public static void loadDataset(String dataDir, List<TFloat32> imageTensors, List<TFloat32> labelTensors) throws IOException {
        classDirs = new File(dataDir).listFiles(File::isDirectory);
        if (classDirs != null) {
            for (File classDir : classDirs) {
                String className = classDir.getName();
                int classLabel = getClassLabel(className);
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

    private static int getClassLabel(String className) {
        return switch (className) {
            case "daisy" -> 0;
            case "dandelion" -> 1;
            case "roses" -> 2;
            case "sunflowers" -> 3;
            case "tulips" -> 4;
            default -> throw new IllegalArgumentException("unknown class: " + className);
        };
    }

    private static TFloat32 preprocessImage(BufferedImage img) {
        int targetWidth = IMAGE_SIZE;
        int targetHeight = IMAGE_SIZE;

        // Convert image to RGB format if it's not already in that format
        if (img.getType() != BufferedImage.TYPE_INT_RGB) {
            BufferedImage rgbImage = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            rgbImage.getGraphics().drawImage(img, 0, 0, null);
            img = rgbImage;
        }

        // Resize image to target dimensions
        BufferedImage resizedImg = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        resizedImg.getGraphics().drawImage(img.getScaledInstance(targetWidth, targetHeight, BufferedImage.SCALE_SMOOTH), 0, 0, null);

        // Prepare array for TensorFlow tensor
        float[][][][] imgArray = new float[1][targetHeight][targetWidth][3];
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resizedImg.getRGB(x, y);
                imgArray[0][y][x][0] = ((rgb >> 16) & 0xFF) / 255.0f; // Red channel
                imgArray[0][y][x][1] = ((rgb >> 8) & 0xFF) / 255.0f;  // Green channel
                imgArray[0][y][x][2] = (rgb & 0xFF) / 255.0f;         // Blue channel
            }
        }
        return TFloat32.tensorOf(StdArrays.ndCopyOf(imgArray));
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

        // Convert imageTensors list to ByteNdArray
        ByteNdArray imageNdArray = NdArrays.ofBytes(Shape.of(imageTensors.size(), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS));
        for (int i = 0; i < imageTensors.size(); i++) {
            imageNdArray.slice(Indices.at(i)).copyFrom(imageTensors.get(i).asRawTensor().data());
        }
        // Convert labelTensors list to FloatNdArray
        ByteNdArray labelNdArray = NdArrays.ofBytes(Shape.of(imageTensors.size()));
        for (int i = 0; i < labelTensors.size(); i++) {
            labelNdArray.slice(Indices.at(i)).copyFrom(labelTensors.get(i).asRawTensor().data());
        }
        train(imageNdArray, labelNdArray);
    }
}
//Fix training process!!!