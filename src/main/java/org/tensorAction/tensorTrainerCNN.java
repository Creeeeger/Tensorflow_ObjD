package org.tensorAction;

import org.object_d.Main_UI;
import org.tensorflow.*;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SavedModel;
import org.tensorflow.framework.losses.Losses;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.BiasAdd;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class tensorTrainerCNN {
    //Main void for testing purposes
    //Usage for loading the dataset and controlling the main process
    public static void main(String[] args) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        String folder = "/Users/gregor/Downloads/flower_photos";
        int numberClasses = 5;
        int imageSize = 32;
        int epochs = 100;
        int batchSize = 200;

        TFloat32[] datasetBatch = loadCocoDataset(folder, batchSize, imageSize, imageSize, 3, numberClasses);
        TFloat32 images = datasetBatch[0];
        TFloat32 labels = datasetBatch[1];
        //debugLoad(images, labels);
        trainModel(images, labels, numberClasses, epochs, imageSize);
    }

    //access void for accessing the program over the trainer
    public static void access(String folder) throws IOException {
        nu.pattern.OpenCV.loadLocally();
        File folderDir = new File(folder);

        int numberClasses = (int) Arrays.stream(Objects.requireNonNull(folderDir.listFiles()))
                .filter(File::isDirectory)
                .count();
        int imageSize = Main_UI.resolution;
        int epochs = Main_UI.epochs;
        int batchSize = Main_UI.batch;

        TFloat32[] datasetBatch = loadCocoDataset(folder, batchSize, imageSize, imageSize, 3, numberClasses);
        TFloat32 images = datasetBatch[0];
        TFloat32 labels = datasetBatch[1];
        trainModel(images, labels, numberClasses, epochs, imageSize);
    }

    //Debug section
    //over here we can check size shape etc. of the images
    public static void debugLoad(TFloat32 images, TFloat32 labels) {
        // Load and preprocess the dataset
        System.out.println("Dataset loaded. Number of photos: " + images.shape().size(0));
        System.out.println(labels.shape());
        System.out.println(images.shape());

        // Iterate over the batch and print the label corresponding to each image
        for (int i = 0; i < images.shape().size(0); i++) {
            // Get the label at index i
            FloatNdArray label = labels.slice(Indices.at(i));

            // Find the class corresponding to the label (assuming one-hot encoding)
            int classIndex = -1;
            for (int j = 0; j < label.shape().size(0); j++) {
                if (label.getFloat(j) == 1.0f) {
                    classIndex = j;
                    break;
                }
            }
            System.out.println("Image index: " + i + ", Class label: " + classIndex);
        }
    }

    public static TFloat32[] loadCocoDataset(String dataDir, int batchSize, int imageHeight, int imageWidth, int numChannels, int numClasses) throws IOException {
        File[] classDirs = new File(dataDir).listFiles(File::isDirectory);
        if (classDirs == null) {
            throw new IOException("No class directories found.");
        }

        // Map class names to integer labels
        Map<String, Integer> classLabelMap = new HashMap<>();
        for (int i = 0; i < classDirs.length; i++) {
            classLabelMap.put(classDirs[i].getName(), i);
        }

        // Initialize arrays for images and labels
        FloatNdArray imageData = NdArrays.ofFloats(Shape.of(batchSize, imageHeight, imageWidth, numChannels));
        FloatNdArray labelData = NdArrays.ofFloats(Shape.of(batchSize, numClasses));

        int index = 0;

        // Loop over class directories and images
        for (File classDir : classDirs) {
            String className = classDir.getName();
            int classLabel = classLabelMap.get(className);
            File[] imageFiles = classDir.listFiles((_, name) -> name.toLowerCase().endsWith(".jpg"));
            if (imageFiles != null) {
                for (File imageFile : imageFiles) {
                    if (index >= batchSize) {
                        break;  // Stop if we've reached the batch size
                    }
                    try {
                        BufferedImage img = ImageIO.read(imageFile);
                        if (img != null) {
                            // Preprocess the image and label
                            float[][][] imageArray = preprocessCocoImage(img, imageHeight, imageWidth);
                            float[] labelArray = preprocessCocoLabel(classLabel, numClasses);

                            // Fill the image tensor
                            for (int i = 0; i < imageHeight; i++) {
                                for (int j = 0; j < imageWidth; j++) {
                                    for (int k = 0; k < numChannels; k++) {
                                        imageData.setFloat(imageArray[i][j][k], index, i, j, k);
                                    }
                                }
                            }

                            // Fill the label tensor
                            for (int l = 0; l < numClasses; l++) {
                                labelData.setFloat(labelArray[l], index, l);
                            }

                            index++;
                        }
                    } catch (IOException e) {
                        System.out.println("Error reading image file: " + imageFile.getName() + " - " + e.getMessage());
                    }
                }
            }
        }

        // Create Tensor<TFloat32> for images and labels
        TFloat32 imageTensor = TFloat32.tensorOf(imageData);
        TFloat32 labelTensor = TFloat32.tensorOf(labelData);

        return new TFloat32[]{imageTensor, labelTensor};  // Return both images and labels
    }

    // Example preprocess image function (adjust this to your needs)
    public static float[][][] preprocessCocoImage(BufferedImage img, int targetHeight, int targetWidth) {
        BufferedImage resizedImage = resizeImage(img, targetHeight, targetWidth);
        float[][][] imageArray = new float[targetHeight][targetWidth][3];  // Assuming 3 channels
        for (int x = 0; x < targetHeight; x++) {
            for (int y = 0; y < targetWidth; y++) {
                int rgb = resizedImage.getRGB(x, y);
                imageArray[x][y][0] = ((rgb >> 16) & 0xFF) / 255.0f;  // Red
                imageArray[x][y][1] = ((rgb >> 8) & 0xFF) / 255.0f;   // Green
                imageArray[x][y][2] = (rgb & 0xFF) / 255.0f;          // Blue
            }
        }
        return imageArray;
    }

    // Example preprocess label function (one-hot encode the class label)
    public static float[] preprocessCocoLabel(int classLabel, int numClasses) {
        float[] labelArray = new float[numClasses];
        labelArray[classLabel] = 1.0f;  // One-hot encoding
        return labelArray;
    }

    // Optional: Resizing image
    public static BufferedImage resizeImage(BufferedImage img, int targetHeight, int targetWidth) {
        // Add your resizing logic here if needed
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(img, 0, 0, targetWidth, targetHeight, null);
        return resizedImage;
    }

    public static Graph CustomGraph(int numClasses, int imageSize) {
        final int NUM_CHANNELS = 3;
        final long SEED = 12345L;

        Graph graph = new Graph();
        Ops tf = Ops.create(graph);

        // Inputs
        Placeholder<TFloat32> input = tf.withName("input").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, imageSize, imageSize, NUM_CHANNELS)));
        Reshape<TFloat32> inputReshaped = tf.reshape(input, tf.array(-1, imageSize, imageSize, NUM_CHANNELS));

        Placeholder<TFloat32> box = tf.withName("box").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, 4)));
        Placeholder<TFloat32> classLabels = tf.withName("labels").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, numClasses)));

        // Input normalization (feature scaling)
        Operand<TFloat32> scaledInput = tf.math.div(
                tf.math.sub(tf.dtypes.cast(inputReshaped, TFloat32.class), tf.constant(127.5f)), tf.constant(255.0f));

        // Build Convolutional Layers + MaxPooling
        Operand<TFloat32> conv1 = buildConvLayer(tf, scaledInput, NUM_CHANNELS, 32);
        Operand<TFloat32> pool1 = buildMaxPoolLayer(tf, conv1);

        Operand<TFloat32> conv2 = buildConvLayer(tf, pool1, 32, 64);
        Operand<TFloat32> pool2 = buildMaxPoolLayer(tf, conv2);

        Operand<TFloat32> conv3 = buildConvLayer(tf, pool2, 64, 128);
        Operand<TFloat32> pool3 = buildMaxPoolLayer(tf, conv3);

        // Flatten for Fully Connected Layers
        Operand<TFloat32> flatten = tf.reshape(pool3, tf.concat(Arrays.asList(
                tf.slice(tf.shape(pool3), tf.array(0), tf.array(1)),
                tf.array(-1)), tf.constant(0)));

        // Fully Connected Layers
        Operand<TFloat32> fc1 = buildFullyConnectedLayer(tf, flatten, imageSize * imageSize * 128 / 64, 512);

        // Classification Output (Softmax)
        Operand<TFloat32> logits = buildFullyConnectedLayer(tf, fc1, 512, numClasses);
        Softmax<TFloat32> classPrediction = tf.withName("class_output").nn.softmax(logits);

        // Bounding Box Output (Regression)
        Operand<TFloat32> boxWeights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(512, 4), TFloat32.class, TruncatedNormal.seed(SEED)),
                tf.constant(0.1f)));
        Operand<TFloat32> boxBiases = tf.variable(tf.fill(tf.array(4), tf.constant(0.1f)));
        Add<TFloat32> boxPrediction = tf.withName("box_output").math.add(tf.linalg.matMul(fc1, boxWeights), boxBiases);

        // Loss Functions: Bounding Box (Huber) + Classification (Cross-Entropy)
        Mean<TFloat32> boxLoss = tf.math.mean(Losses.huber(tf, box, boxPrediction, 1.0f), tf.constant(0));
        // Compute softmax cross-entropy loss
        SoftmaxCrossEntropyWithLogits<TFloat32> crossEntropy = tf.nn.softmaxCrossEntropyWithLogits(logits, classLabels);
        Mean<TFloat32> classLoss = tf.math.mean(crossEntropy.loss(), tf.constant(0));

        // Regularization (L2 Loss)
        Add<TFloat32> regularizers = tf.math.add(tf.nn.l2Loss(fc1), tf.nn.l2Loss(logits));
        Add<TFloat32> totalLoss = tf.withName("totalLoss").math.add(tf.math.add(boxLoss, classLoss),
                tf.math.mul(regularizers, tf.constant(5e-4f)));

        // Optimizer (Adam)
        Optimizer optimizer = new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f);
        optimizer.minimize(totalLoss, "train");

        return graph;
    }

    // Combined method for Convolutional + Max Pooling layers
    private static Operand<TFloat32> buildConvLayer(Ops tf, Operand<TFloat32> input, int inputChannels, int outputChannels) {
        Operand<TFloat32> convWeights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(5, 5, inputChannels, outputChannels),
                TFloat32.class, TruncatedNormal.seed(12345L)), tf.constant(0.1f)));
        Operand<TFloat32> convBiases = tf.variable(tf.fill(tf.array(outputChannels), tf.constant(0.0f)));
        Conv2d<TFloat32> conv = tf.nn.conv2d(input, convWeights, Arrays.asList(1L, 1L, 1L, 1L), "SAME");
        BiasAdd<TFloat32> biasAdd = tf.nn.biasAdd(conv, convBiases);
        return tf.nn.relu(biasAdd);
    }

    private static Operand<TFloat32> buildMaxPoolLayer(Ops tf, Operand<TFloat32> input) {
        return tf.nn.maxPool(input, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1), "SAME");
    }

    // Combined method for Fully Connected Layers
    private static Operand<TFloat32> buildFullyConnectedLayer(Ops tf, Operand<TFloat32> input, int inputUnits, int outputUnits) {
        Operand<TFloat32> weights = tf.variable(tf.math.mul(
                tf.random.truncatedNormal(tf.array(inputUnits, outputUnits), TFloat32.class, TruncatedNormal.seed(12345L)),
                tf.constant(0.1f)));
        Operand<TFloat32> biases = tf.variable(tf.fill(tf.array(outputUnits), tf.constant(0.1f)));
        Operand<TFloat32> dense = tf.math.add(tf.linalg.matMul(input, weights), biases);
        return tf.nn.relu(dense);
    }

    public static TFloat32 generateSyntheticboxes(int batchSize) {
        // Create a tensor of shape [batchSize, 4] with synthetic bounding box data
        float[][] boxData = new float[batchSize][4];
        for (int i = 0; i < batchSize; i++) {
            boxData[i][0] = (float) Math.random(); // x_min
            boxData[i][1] = (float) Math.random(); // y_min
            boxData[i][2] = (float) Math.random(); // x_max
            boxData[i][3] = (float) Math.random(); // y_max
        }
        return TFloat32.tensorOf(StdArrays.ndCopyOf(boxData));
    }

    public static void trainModel(TFloat32 images, TFloat32 labels, int numClasses, int epochs, int imageSize) throws IOException {
        // Create the graph and initialize variables
        try (Graph graph = CustomGraph(numClasses, imageSize);
             Session session = new Session(graph)) {

            // Setup Adam Optimizer
            new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f);

            TFloat32 boxTensor = generateSyntheticboxes((int) images.shape().size(0));

            Result outputs = null;

            for (int epoch = 0; epoch < epochs; epoch++) {
                // Run the session to compute the loss and optimize
                Session.Runner runner = session.runner()
                        .feed("box", boxTensor)
                        .feed("input", images)
                        .feed("labels", labels)
                        .addTarget("train");

                // Fetch the loss values for monitoring
                outputs = runner
                        .fetch("box_output")
                        .fetch("class_output")
                        .fetch("totalLoss")
                        .run();

                // Print loss information (optional for debugging)
                TFloat32 lossTensor = (TFloat32) outputs.get(0);
                TFloat32 lossTensor1 = (TFloat32) outputs.get(1);
                TFloat32 lossTensor2 = (TFloat32) outputs.get(2);

                System.out.printf("Loss at epoch %d: %-10.6f %-10.6f %-10.6f%n", epoch, lossTensor.getFloat(), lossTensor1.getFloat(), lossTensor2.getFloat());

                // Close output tensors
                for (Map.Entry<String, Tensor> tensor : outputs) {
                    tensor.getValue().close();
                }
            }

            System.out.println("Training completed.");
            validate(session, images, labels, numClasses, outputs);

            saveModel(graph, session, Paths.get(Paths.get("").toAbsolutePath().toString()).getParent().toString());

            System.out.println("Model saved");
        }
    }

    // Method to test the model and compute accuracy and confusion matrix
    public static void validate(Session session, TFloat32 image, TFloat32 label, int numClasses, Result outputs) {
        int correctCount = 0;
        int[][] confusionMatrix = new int[numClasses][numClasses];

        // Iterate over each test data sample
        for (int i = 0; i < image.shape().size(0); i++) {
            FloatNdArray imageND = image.slice(Indices.slice(i, i+1));
            TFloat32 imageSingle = TFloat32.tensorOf(NdArrays.ofFloats(imageND.shape()));

            // Perform prediction
            TFloat32 outputTensor = (TFloat32) session.runner()
                    .feed("input", imageSingle)
                    .fetch("class_output")
                    .run()
                    .get(0);

            // Convert trueLabelTensor to an integer
            int trueLabel = argmaxLabel(label, i);

            // Get predicted label
            int predictedLabel = 1;

            // Compare prediction with true label
            if (predictedLabel == trueLabel) {
                correctCount++;
            }

            // Update confusion matrix
            confusionMatrix[trueLabel][predictedLabel]++;

        }

        // Assuming classPredictionTensor is the tensor that contains the predicted probabilities (softmax output)
        TFloat32 classPredictionTensor = (TFloat32) outputs.get(1);  // Fetch the class predictions (softmax output)
        // Get the shape of the tensor to iterate over the batch and number of classes
        long batchSize = classPredictionTensor.shape().size(0);  // Number of images in the batch
        long Classes = classPredictionTensor.shape().size(1);  // Number of classes

        int[] predictedLabels = new int[(int) batchSize];  // Array to store predicted labels for each image

        for (int i = 0; i < batchSize; i++) {
            float maxProb = -1.0f;  // Track the maximum probability for each image
            int labell = -1;  // Track the index (class) with the maximum probability

            for (int j = 0; j < Classes; j++) {
                float prob = classPredictionTensor.getFloat(i, j);  // Get the probability for class j of image i
                if (prob > maxProb) {
                    maxProb = prob;  // Update the maximum probability
                    labell = j;  // Update the predicted label (class index)
                }
            }

            predictedLabels[i] = labell;  // Store the predicted label for image i
        }

        // Now, predictedLabels contains the predicted class for each image in the batch
        for (int i = 0; i < predictedLabels.length; i++) {
            System.out.println("Predicted label for image " + i + ": " + predictedLabels[i] + " True label: " + argmaxLabel(label, i));
        }

        // Print accuracy and confusion matrix
        float accuracy = (float) correctCount / image.shape().size(0);
        System.out.println("Final accuracy: " + accuracy);
        System.out.println(getStringBuilder(confusionMatrix));
    }

    // Method to build the confusion matrix as a string
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

    // Method to get the index of the maximum value in the tensor
    public static int argmaxLabel(TFloat32 tensor, int iteration) {
        FloatNdArray label = tensor.slice(Indices.at(iteration));

        // Find the class corresponding to the label (assuming one-hot encoding)
        int classIndex = -1;
        for (int i = 0; i < label.shape().size(0); i++) {
            if (label.getFloat(i) == 1.0f) {
                classIndex = i;
                break;
            }
        }
        return classIndex;
    }

    public static void saveModel(Graph graph, Session session, String exportDir) throws IOException {
        // Create the directories if they don't exist
        Files.createDirectories(Paths.get(exportDir, "model"));
        Files.createDirectories(Paths.get(exportDir, "model", "variables"));

        MetaGraphDef.Builder metaGraphDefBuilder = MetaGraphDef.newBuilder();
        metaGraphDefBuilder.setGraphDef(GraphDef.parseFrom(graph.toGraphDef().toByteArray()));

        // Create MetaInfoDef and add serve tag
        MetaGraphDef.MetaInfoDef.Builder metaInfoDefBuilder = MetaGraphDef.MetaInfoDef.newBuilder();
        metaInfoDefBuilder.addTags("serve");  // Use addTags method to add a tag to the list
        metaGraphDefBuilder.setMetaInfoDef(metaInfoDefBuilder.build());  // Set the built MetaInfoDef

        SavedModel.Builder builder = SavedModel.newBuilder();
        builder.addMetaGraphs(metaGraphDefBuilder);

        // Save session variables to model/variables directory
        session.save(exportDir + "/model/variables/variables");

        // Write the MetaGraphDef to saved_model.pb in the model directory
        Files.write(Paths.get(exportDir, "model", "saved_model.pb"), builder.build().toByteArray());
        System.out.println("Model saved to " + exportDir + "/model/saved_model.pb");
    }
}