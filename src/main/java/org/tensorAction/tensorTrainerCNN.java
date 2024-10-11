package org.tensorAction;

import org.object_d.Main_UI;
import org.tensorflow.*;
import org.tensorflow.framework.*;
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
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.*;

public class tensorTrainerCNN extends JFrame {
    static int numberClasses, epochs; // Declare variables for the number of classes and epochs (training iterations)
    static float maxLoss = Main_UI.learning_rate; // Initialize maxLoss with the learning rate value from the Main_UI class
    static int[][] confusionMatrix; // Declare a 2D array for the confusion matrix
    static List<Float> boxLossValues = new ArrayList<>();  // List to store loss values for bounding boxes
    static List<Float> classLossValues = new ArrayList<>();  // List to store loss values for class predictions
    static List<Float> totalLossValues = new ArrayList<>();  // List to store total loss values (combined)
    static JTextArea textArea; // Declare a text area for displaying output as well as the matrix
    static JLabel accuracy_label; // Label to display the accuracy of the model
    JPanel box_loss_panel, class_loss_panel, total_loss_panel, confusion_matrix_panel; // Panels to display loss values and confusion matrix

    public tensorTrainerCNN() {
        setLayout(new GridLayout(4, 1, 10, 10)); // Set layout for the panel, a 4-row grid with spacing between elements

        // Initialize and configure the box loss panel
        box_loss_panel = new JPanel();
        box_loss_panel.setBorder(BorderFactory.createTitledBorder("Box loss")); // Set a titled border for the panel
        box_loss_graph boxLossGraph = new box_loss_graph(); // Create an instance of the graph for box loss
        box_loss_panel.add(boxLossGraph); // Add the graph to the panel

        // Initialize and configure the class loss panel
        class_loss_panel = new JPanel();
        class_loss_panel.setBorder(BorderFactory.createTitledBorder("Class loss")); // Set a titled border for the panel
        class_loss_graph classLossGraph = new class_loss_graph(); // Create an instance of the graph for class loss
        class_loss_panel.add(classLossGraph); // Add the graph to the panel

        // Initialize and configure the total loss panel
        total_loss_panel = new JPanel();
        total_loss_panel.setBorder(BorderFactory.createTitledBorder("Total loss")); // Set a titled border for the panel
        total_loss_graph totalLossGraph = new total_loss_graph(); // Create an instance of the graph for total loss
        total_loss_panel.add(totalLossGraph); // Add the graph to the panel

        // Initialize and configure the confusion matrix panel
        confusion_matrix_panel = new JPanel();
        confusion_matrix_panel.setBorder(BorderFactory.createTitledBorder("Confusion matrix")); // Set a titled border

        // Add the panels to the layout
        add(box_loss_panel);
        add(class_loss_panel);
        add(total_loss_panel);
        add(confusion_matrix_panel);

        // Initialize the text area for the confusion matrix and the accuracy label
        // Set text area size dynamically based on the number of classes
        // + 2 because the number row is one and the counting process requires one more
        textArea = new JTextArea(numberClasses + 2, numberClasses + 2);
        accuracy_label = new JLabel("Final accuracy: ..."); // Label to display final accuracy

        // Add the accuracy label and text area to the confusion matrix panel
        confusion_matrix_panel.add(accuracy_label);
        confusion_matrix_panel.add(textArea);
    }

    // Method to access the program for training a model using the specified folder
    public static void access(String folder) throws IOException {
        // Load the OpenCV library locally this is the new method since the native library method doesn't work anymore
        nu.pattern.OpenCV.loadLocally();

        // Create a File object pointing to the provided folder directory
        File folderDir = new File(folder);

        // Count the number of subdirectories (representing different classes) in the folder images must be grouped for this in the folder depending on classes
        numberClasses = (int) Arrays.stream(Objects.requireNonNull(folderDir.listFiles()))
                .filter(File::isDirectory)
                .count();

        // Retrieve training configuration from Main_UI settings
        int imageSize = Main_UI.resolution; // Get the image resolution
        epochs = Main_UI.epochs; // Get the number of training epochs
        int batchSize = Main_UI.batch_size; // Get the batch size for training

        // Check if the folder contains no subdirectories (i.e., no grouped images)
        if (numberClasses == 0) {
            throw new RuntimeException("You can't use a folder without grouped images!"); // Throw an error if no classes found
        }

        // Load the dataset into batches with specified parameters
        TFloat32[] datasetBatch = loadDataset(folder, batchSize, imageSize, imageSize, 3, numberClasses);
        TFloat32 images = datasetBatch[0]; // Extract the image batch
        TFloat32 labels = datasetBatch[1]; // Extract the corresponding labels batch

        // Train the model with the loaded dataset, number of classes, epochs, and image size
        trainModel(images, labels, numberClasses, epochs, imageSize);
    }

    // Method to load a dataset from a specified directory, preprocess images, and prepare them for training
    public static TFloat32[] loadDataset(String dataDir, int batchSize, int imageHeight, int imageWidth, int numChannels, int numClasses) throws IOException {
        // Get all class directories from the specified data directory
        File[] classDirs = new File(dataDir).listFiles(File::isDirectory);

        // Check if no class directories are found and throw an exception if so
        if (classDirs == null) {
            throw new IOException("No class directories found.");
        }

        // Create a mapping of class names to their corresponding integer labels so that each label is linked to an int value
        Map<String, Integer> classLabelMap = new HashMap<>();
        for (int i = 0; i < classDirs.length; i++) {
            classLabelMap.put(classDirs[i].getName(), i);  // Associate class name with its index
        }

        // Initialize arrays for holding image and label data
        FloatNdArray imageData = NdArrays.ofFloats(Shape.of(batchSize, imageHeight, imageWidth, numChannels)); // FloatNdArray for image data
        FloatNdArray labelData = NdArrays.ofFloats(Shape.of(batchSize, numClasses)); // FloatNdArray for label data

        int index = 0;  // Index to track the number of images processed

        // Loop through each class directory to process the images
        for (File classDir : classDirs) {
            String className = classDir.getName();  // Get the current class name
            int classLabel = classLabelMap.get(className);  // Retrieve the corresponding class label
            File[] imageFiles = classDir.listFiles((_, name) -> name.toLowerCase().endsWith(".jpg")); // Get all JPG images in the class directory and convert them into lower case

            // Check if there are image files to process
            if (imageFiles != null) {
                // Loop through each image file in the current class directory
                for (File imageFile : imageFiles) {
                    // Stop processing if the batch size limit is reached
                    if (index >= batchSize) {
                        break;
                    }
                    try {
                        // Read the image file into a BufferedImage object
                        BufferedImage img = ImageIO.read(imageFile);
                        if (img != null) {
                            // Preprocess the image to resize and normalize it
                            float[][][] imageArray = preprocessImage(img, imageHeight, imageWidth);

                            // Create the one-hot encoded label for the current class
                            float[] labelArray = preprocessLabel(classLabel, numClasses);

                            // Fill the image tensor with pixel values from the preprocessed image array
                            for (int i = 0; i < imageHeight; i++) { // Iterate over each row of the image
                                for (int j = 0; j < imageWidth; j++) { // Iterate over each column of the image
                                    for (int k = 0; k < numChannels; k++) { // Iterate over each color channel (e.g., R, G, B)
                                        // Set the pixel value at the specified index in the image tensor
                                        // imageArray[i][j][k] contains the normalized pixel value for the pixel at (i, j) for channel k
                                        imageData.setFloat(imageArray[i][j][k], index, i, j, k);
                                    }
                                }
                            }

                            // Fill the label tensor with the one-hot encoded label
                            for (int l = 0; l < numClasses; l++) {
                                labelData.setFloat(labelArray[l], index, l);
                            }

                            index++;  // Increment the index after processing an image
                        }
                    } catch (IOException e) {
                        // Handle potential IO exceptions and print an error message
                        System.out.println("Error reading image file: " + imageFile.getName() + " - " + e.getMessage());
                    }
                }
            }
        }

        // Create tensors for images and labels from the filled FloatNdArray objects
        TFloat32 imageTensor = TFloat32.tensorOf(imageData);  // Tensor for image data
        TFloat32 labelTensor = TFloat32.tensorOf(labelData);  // Tensor for label data

        // Return an array containing both the image tensor and label tensor
        return new TFloat32[]{imageTensor, labelTensor};  // Return both images and labels
    }

    // Method to preprocess an image by resizing it and converting it to a normalized float array
    public static float[][][] preprocessImage(BufferedImage img, int targetHeight, int targetWidth) {
        // Resize the input image to the target dimensions
        BufferedImage resizedImage = resizeImage(img, targetHeight, targetWidth);

        // Create a 3D array to hold the pixel values for the image
        // Dimensions: height x width x color channels (assuming 3 channels for RGB)
        float[][][] imageArray = new float[targetHeight][targetWidth][3];  // 3 channels: Red, Green, Blue

        // Loop through each pixel in the resized image
        for (int x = 0; x < targetHeight; x++) { // Iterate over each row (height)
            for (int y = 0; y < targetWidth; y++) { // Iterate over each column (width)
                // Retrieve the RGB value of the current pixel at position (x, y)
                int rgb = resizedImage.getRGB(x, y);

                // Extract the individual color components from the RGB value
                // Normalize the Red component to the range [0, 1]
                imageArray[x][y][0] = ((rgb >> 16) & 0xFF) / 255.0f;  // Red channel (bits 16-23)

                // Normalize the Green component to the range [0, 1]
                imageArray[x][y][1] = ((rgb >> 8) & 0xFF) / 255.0f;   // Green channel (bits 8-15)

                // Normalize the Blue component to the range [0, 1]
                imageArray[x][y][2] = (rgb & 0xFF) / 255.0f;          // Blue channel (bits 0-7)
            }
        }

        // Return the 3D array containing the normalized pixel values
        return imageArray;
    }

    // Method to preprocess a label for a given class label and total number of classes
    public static float[] preprocessLabel(int classLabel, int numClasses) {
        // Create an array to hold the one-hot encoded label, initialized to zero
        float[] labelArray = new float[numClasses];

        // Set the position corresponding to the class label to 1.0 for one-hot encoding
        labelArray[classLabel] = 1.0f;  // One-hot encoding

        // Return the one-hot encoded label array
        return labelArray;
    }

    // Method to resize a given image to specified dimensions
    public static BufferedImage resizeImage(BufferedImage img, int targetHeight, int targetWidth) {
        // Create a new BufferedImage with the target dimensions and RGB color model
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);

        // Draw the original image onto the new resized image with the specified dimensions
        resizedImage.getGraphics().drawImage(img, 0, 0, targetWidth, targetHeight, null);

        // Return the resized image
        return resizedImage;
    }

    public static Graph Graph(int numClasses, int imageSize) {
        // Define constants for the number of channels and random seed for initialization
        final int NUM_CHANNELS = 3; // RGB image
        final long SEED = 12345L; // Seed for random number generation

        // Create a new computation graph
        Graph graph = new Graph();
        Ops tf = Ops.create(graph); // TensorFlow operations instance

        // Input placeholders
        // Input tensor for image data, shape: [batch_size, imageSize, imageSize, NUM_CHANNELS]
        Placeholder<TFloat32> input = tf.withName("input").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, imageSize, imageSize, NUM_CHANNELS)));
        // Reshape input tensor if necessary
        Reshape<TFloat32> inputReshaped = tf.reshape(input, tf.array(-1, imageSize, imageSize, NUM_CHANNELS));

        // Placeholder for bounding box coordinates and class labels
        Placeholder<TFloat32> box = tf.withName("box").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, 4))); // shape for bounding boxes
        Placeholder<TFloat32> classLabels = tf.withName("labels").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, numClasses))); // shape for class labels

        // Input normalization (feature scaling)
        // Scale pixel values from [0, 255] to [-1, 1] for better training convergence
        Operand<TFloat32> scaledInput = tf.math.div(
                tf.math.sub(tf.dtypes.cast(inputReshaped, TFloat32.class), tf.constant(127.5f)), tf.constant(255.0f));

        // Build Convolutional Layers followed by Max Pooling layers
        Operand<TFloat32> conv1 = buildConvLayer(tf, scaledInput, NUM_CHANNELS, 32); // First convolution layer
        Operand<TFloat32> pool1 = buildMaxPoolLayer(tf, conv1); // First max pooling layer

        Operand<TFloat32> conv2 = buildConvLayer(tf, pool1, 32, 64); // Second convolution layer
        Operand<TFloat32> pool2 = buildMaxPoolLayer(tf, conv2); // Second max pooling layer

        Operand<TFloat32> conv3 = buildConvLayer(tf, pool2, 64, 128); // Third convolution layer
        Operand<TFloat32> pool3 = buildMaxPoolLayer(tf, conv3); // Third max pooling layer

        // Flatten the output from the last pooling layer to feed into fully connected layers
        Operand<TFloat32> flatten = tf.reshape(pool3, tf.concat(Arrays.asList(
                tf.slice(tf.shape(pool3), tf.array(0), tf.array(1)), // Keep batch size
                tf.array(-1)), tf.constant(0))); // Flatten other dimensions

        // Fully Connected Layers
        Operand<TFloat32> fc1 = buildFullyConnectedLayer(tf, flatten, imageSize * imageSize * 128 / 64, 512); // First fully connected layer

        // Classification Output using Softmax activation
        Operand<TFloat32> logits = buildFullyConnectedLayer(tf, fc1, 512, numClasses); // Fully connected layer for class logits
        tf.withName("class_output").nn.softmax(logits); // Apply softmax to logits for class probabilities

        // Bounding Box Output for regression
        // Initialize weights and biases for bounding box predictions
        Operand<TFloat32> boxWeights = tf.variable(tf.math.mul(tf.random.truncatedNormal(tf.array(512, 4), TFloat32.class, TruncatedNormal.seed(SEED)),
                tf.constant(0.1f))); // Weights for bounding box regression
        Operand<TFloat32> boxBiases = tf.variable(tf.fill(tf.array(4), tf.constant(0.1f))); // Biases for bounding box regression
        Add<TFloat32> boxPrediction = tf.withName("box_output").math.add(tf.linalg.matMul(fc1, boxWeights), boxBiases); // Box prediction calculation

        // Loss Functions: Compute losses for bounding boxes and classification
        Mean<TFloat32> boxLoss = tf.math.mean(Losses.huber(tf, box, boxPrediction, 1.0f), tf.constant(0)); // Huber loss for bounding boxes
        // Compute softmax cross-entropy loss for classification
        SoftmaxCrossEntropyWithLogits<TFloat32> crossEntropy = tf.nn.softmaxCrossEntropyWithLogits(logits, classLabels);
        Mean<TFloat32> classLoss = tf.math.mean(crossEntropy.loss(), tf.constant(0)); // Mean cross-entropy loss

        // Regularization (L2 Loss) to prevent overfitting
        Add<TFloat32> regularizers = tf.math.add(tf.nn.l2Loss(fc1), tf.nn.l2Loss(logits)); // L2 loss for fully connected layer and logits
        // Compute total loss as the sum of box loss, class loss, and regularization
        Add<TFloat32> totalLoss = tf.withName("totalLoss").math.add(tf.math.add(boxLoss, classLoss),
                tf.math.mul(regularizers, tf.constant(5e-4f))); // Scale regularization term

        // Optimizer (Adam) for minimizing the total loss
        Optimizer optimizer = new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f); // Create an Adam optimizer
        optimizer.minimize(totalLoss, "train"); // Add minimization operation to the optimizer

        return graph; // Return the constructed computation graph
    }

    // Method to build a Convolutional Layer followed by ReLU activation
    private static Operand<TFloat32> buildConvLayer(Ops tf, Operand<TFloat32> input, int inputChannels, int outputChannels) {
        // Create the filter with a 5x5 kernel, inputChannels -> depth, and outputChannels -> number of filters
        // Truncated normal distribution is used -> limiting extreme values
        Operand<TFloat32> convWeights = tf.variable(tf.math.mul(
                tf.random.truncatedNormal(tf.array(5, 5, inputChannels, outputChannels),
                        TFloat32.class, TruncatedNormal.seed(12345L)), // Ensure reproducibility with a fixed seed
                tf.constant(0.1f))); // Scale the randomly initialized weights by 0.1

        // Create a bias for the operation, initialized to 0 for each output channel
        Operand<TFloat32> convBiases = tf.variable(tf.fill(tf.array(outputChannels), tf.constant(0.0f)));

        // Perform the 2D convolution operation
        // "SAME" padding ensures that the output has the same dimensions as the input
        Conv2d<TFloat32> conv = tf.nn.conv2d(input, convWeights, Arrays.asList(1L, 1L, 1L, 1L), "SAME");

        // Add the biases to the convolution result, shifting the values for each output channel
        BiasAdd<TFloat32> biasAdd = tf.nn.biasAdd(conv, convBiases);

        // Apply the ReLU activation function to introduce non-linearity. ReLU sets negative values to 0 and keeps positive values as they are
        return tf.nn.relu(biasAdd); // ->learn more complex patterns
    }

    // Method to build a Max Pooling Layer for scaling image down
    private static Operand<TFloat32> buildMaxPoolLayer(Ops tf, Operand<TFloat32> input) {
        // Apply max pooling with a 2x2 filter which halves the height and width of the input
        // "SAME" padding ensures that the output size is reduced evenly
        return tf.nn.maxPool(input, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1), "SAME");
        // Max Pooling helps keeping important features while reducing computational complexity
    }

    // Method to build a Fully Connected Layer + ReLU
    private static Operand<TFloat32> buildFullyConnectedLayer(Ops tf, Operand<TFloat32> input, int inputUnits, int outputUnits) {
        // Initialize the weights matrix for the fully connected layer with inputUnits (number of input features) and outputUnits (number of neurons)
        Operand<TFloat32> weights = tf.variable(tf.math.mul(
                tf.random.truncatedNormal(tf.array(inputUnits, outputUnits), TFloat32.class, TruncatedNormal.seed(12345L)),
                tf.constant(0.1f)));

        // Initialize biases for each output unit (neuron), set to a small positive value (0.1) to avoid "dead neurons"
        Operand<TFloat32> biases = tf.variable(tf.fill(tf.array(outputUnits), tf.constant(0.1f)));

        // Perform matrix multiplication between the input and weights, which combines features across neurons
        // Add biases to the result of the matrix multiplication, shifting the values before applying activation
        Operand<TFloat32> dense = tf.math.add(tf.linalg.matMul(input, weights), biases);

        // Apply the ReLU activation function to the result of the fully connected layer to introduce non-linearity
        return tf.nn.relu(dense);
    }

    public static TFloat32 generate_Synthetic_boxes(int batchSize) {
        // Create a 2D float array to hold the bounding box data for each image in the batch
        float[][] boxData = new float[batchSize][4];

        // Loop through each item in the batch to generate random synthetic bounding boxes
        for (int i = 0; i < batchSize; i++) {
            // Assign random float values (0 to 1) for the bounding box coordinates
            boxData[i][0] = (float) Math.random(); // x_min (left)
            boxData[i][1] = (float) Math.random(); // y_min (top)
            boxData[i][2] = (float) Math.random(); // x_max (right)
            boxData[i][3] = (float) Math.random(); // y_max (bottom)
        }

        // Convert the 2D array into a TFloat32 tensor and return it
        return TFloat32.tensorOf(StdArrays.ndCopyOf(boxData));
    }

    public static void trainModel(TFloat32 images, TFloat32 labels, int numClasses, int epochs, int imageSize) throws IOException {
        // Initialize and display the live training analysis GUI window
        tensorTrainerCNN gui = new tensorTrainerCNN();
        gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set the window to hide on close
        gui.setTitle("Live training analysis"); // Set the window title
        gui.setVisible(true); // Make the window visible
        gui.setLocation(100, 10); // Position the window on the screen
        gui.pack(); // Adjust the window to fit its content

        // Create a new computation graph and session
        try (Graph graph = Graph(numClasses, imageSize);
             Session session = new Session(graph)) {

            // Initialize the Adam optimizer
            new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f);

            // Generate synthetic bounding box data for the batch of images
            TFloat32 boxTensor = generate_Synthetic_boxes((int) images.shape().get(0));

            Result outputs = null;

            // Loop over the specified number of training epochs
            for (int epoch = 0; epoch < epochs; epoch++) {
                // Run the session, feeding the images, labels, and synthetic box data, and target training
                Session.Runner runner = session.runner()
                        .feed("box", boxTensor)  // Feed synthetic bounding box data
                        .feed("input", images)   // Feed image data
                        .feed("labels", labels)  // Feed label data
                        .addTarget("train");     // Target the "train" operation for optimization

                // Fetch the loss values for different components (box loss, class loss, total loss)
                outputs = runner
                        .fetch("box_output")
                        .fetch("class_output")
                        .fetch("totalLoss")
                        .run();  // Run the session and collect the results

                // Get the loss values from the fetched outputs
                TFloat32 lossTensor = (TFloat32) outputs.get(0);   // Box loss
                TFloat32 lossTensor1 = (TFloat32) outputs.get(1);  // Class loss
                TFloat32 lossTensor2 = (TFloat32) outputs.get(2);  // Total loss

                // Print the loss values for the current epoch
                System.out.printf("Loss at epoch %d: %-10.6f %-10.6f %-10.6f%n", epoch, lossTensor.getFloat(), lossTensor1.getFloat(), lossTensor2.getFloat());

                // Update the GUI with the new loss values for live visualization
                gui.updateLossValues(lossTensor.getFloat(), lossTensor1.getFloat(), lossTensor2.getFloat());

                // Close output tensors to free resources
                for (Map.Entry<String, Tensor> tensor : outputs) {
                    tensor.getValue().close();
                }
            }

            // Print completion message after training is done
            System.out.println("Training completed.");

            // Validate the model using the labels and outputs after training
            validate(labels, numClasses, Objects.requireNonNull(outputs));

            // Save the trained model to the specified directory
            saveModel(graph, session, Paths.get(Paths.get("").toAbsolutePath().toString()).getParent().toString());

            // Print message confirming model save
            System.out.println("Model saved");
        }
    }

    // Method to test the model and compute accuracy and confusion matrix
    public static void validate(TFloat32 label, int numClasses, Result outputs) {
        int correctCount = 0;  // Variable to track the number of correct predictions
        confusionMatrix = new int[numClasses][numClasses];  // Initialize confusion matrix with the size of the classes

        // Retrieve the class prediction tensor (softmax output for class probabilities)
        TFloat32 classPredictionTensor = (TFloat32) outputs.get(1);

        // Get the batch size (number of images) and the number of classes from the tensor's shape
        long batchSize = classPredictionTensor.shape().get(0);  // Number of images
        long Classes = classPredictionTensor.shape().get(1);  // Number of possible classes

        int[] predictedLabels = new int[(int) batchSize];  // Array to store predicted labels for each image in the batch

        // Iterate over each image in the batch
        for (int i = 0; i < batchSize; i++) {
            float maxProb = -1.0f;  // Variable to track the maximum probability for the current image
            int softmax_label = -1;  // Variable to store the predicted class (index of maximum probability)

            // Iterate over each class to find the class with the highest probability (softmax output)
            for (int j = 0; j < Classes; j++) {
                float prob = classPredictionTensor.getFloat(i, j);  // Get the probability for class j of image i
                if (prob > maxProb) {
                    maxProb = prob;  // Update max probability
                    softmax_label = j;  // Update predicted label (class index)
                }
            }

            predictedLabels[i] = softmax_label;  // Store the predicted label for image i
        }

        // Iterate over the predicted labels and true labels for accuracy calculation and confusion matrix update
        for (int i = 0; i < predictedLabels.length; i++) {
            System.out.println("Predicted label for image " + i + ": " + predictedLabels[i] + " True label: " + argmaxLabel(label, i));

            // Check if the prediction matches the true label
            if (predictedLabels[i] == argmaxLabel(label, i)) {
                correctCount++;  // Increment the correct count if the prediction is correct
            }

            // Update the confusion matrix (true label vs. predicted label)
            confusionMatrix[argmaxLabel(label, i)][predictedLabels[i]]++;
        }

        // Calculate the overall accuracy of the model
        float accuracy = (float) correctCount / classPredictionTensor.shape().get(0);
        System.out.println("Final accuracy: " + accuracy);

        // Print and display the confusion matrix
        System.out.println(getStringBuilder(confusionMatrix));
        textArea.setText(getStringBuilder(confusionMatrix).toString());
        accuracy_label.setText("Final accuracy: " + accuracy);
    }

    // Method to build the confusion matrix as a formatted string
    private static StringBuilder getStringBuilder(int[][] confusionMatrix) {
        StringBuilder sb = new StringBuilder();  // StringBuilder to hold the formatted confusion matrix string

        // Append column headers ("Label" and class numbers)
        sb.append("Label");
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%1$5s", "" + i));  // Append each class number with right alignment
        }
        sb.append("\n");  // Move to the next line

        // Loop through each row of the confusion matrix
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%1$5s", "" + i));  // Append the row label (class number)

            // Append the confusion matrix values for the current row
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                sb.append(String.format("%1$5s", "" + confusionMatrix[i][j]));  // Append each value with right alignment
            }
            sb.append("\n");  // Move to the next line after the row is printed
        }
        return sb;  // Return the formatted confusion matrix as a StringBuilder
    }

    // Method to get the index of the maximum value (class label) in the tensor for a given iteration
    public static int argmaxLabel(TFloat32 tensor, int iteration) {
        // Extract the label for the specified iteration (slice of the tensor)
        FloatNdArray label = tensor.slice(Indices.at(iteration));

        // Variable to store the index of the class with a value of 1.0 (indicating the correct class)
        int classIndex = 0;

        // Loop through the label array to find the index where the value is 1.0
        for (int i = 0; i < label.shape().get(0); i++) {
            if (label.getFloat(i) == 1.0f) {  // Check if the value at index i is 1.0 (one-hot encoding)
                classIndex = i;  // Set classIndex to the current index
                break;  // Exit the loop once the correct class is found
            }
        }

        // Return the index of the correct class label
        return classIndex;
    }

    public static void saveModel(Graph graph, Session session, String exportDir) throws IOException {
        // Create the main model directory and the variables subdirectory
        Path modelDir = Paths.get(exportDir, "model");
        Files.createDirectories(modelDir); // Create the model directory if it doesn't exist
        Files.createDirectories(modelDir.resolve("variables")); // Create the variables subdirectory

        // Initialize a MetaGraphDef to store the graph definition and associated metadata
        MetaGraphDef.Builder metaGraphDefBuilder = MetaGraphDef.newBuilder();
        // Convert the graph to its byte representation and add it to the MetaGraphDef
        metaGraphDefBuilder.setGraphDef(GraphDef.parseFrom(graph.toGraphDef().toByteArray()));

        // Create a MetaInfoDef to store additional information about the graph
        MetaGraphDef.MetaInfoDef.Builder metaInfoDefBuilder = MetaGraphDef.MetaInfoDef.newBuilder();
        metaInfoDefBuilder.addTags("serve"); // Tag for serving the model
        metaGraphDefBuilder.setMetaInfoDef(metaInfoDefBuilder.build()); // Add the MetaInfoDef to the MetaGraphDef

        // Define a SignatureDef for the model's input and output tensors
        SignatureDef.Builder signatureDefBuilder = SignatureDef.newBuilder();

        // Create an input tensor signature to define the model's input shape and type
        TensorInfo inputTensorInfo = TensorInfo.newBuilder()
                .setDtype(DataType.DT_FLOAT) // Set the data type to float
                .setTensorShape(TensorShapeProto.newBuilder()
                        .addDim(TensorShapeProto.Dim.newBuilder().setSize(-1)) // Batch size (dynamic)
                        .addDim(TensorShapeProto.Dim.newBuilder().setSize(-1)) // Height (dynamic)
                        .addDim(TensorShapeProto.Dim.newBuilder().setSize(-1)) // Width (dynamic)
                        .addDim(TensorShapeProto.Dim.newBuilder().setSize(-1)) // Channels (dynamic)
                )
                .setName("input") // Name of the input tensor
                .build();
        signatureDefBuilder.putInputs("input", inputTensorInfo); // Add input tensor info to the signature

        // Create output tensor signatures for class predictions and bounding box predictions
        TensorInfo classOutputTensorInfo = TensorInfo.newBuilder()
                .setDtype(DataType.DT_FLOAT) // Set the data type to float
                .setTensorShape(TensorShapeProto.newBuilder().addDim(TensorShapeProto.Dim.newBuilder().setSize(-1))) // Dynamic output size
                .setName("class_output") // Name of the class output tensor
                .build();
        signatureDefBuilder.putOutputs("class_output", classOutputTensorInfo); // Add to signature

        TensorInfo boxOutputTensorInfo = TensorInfo.newBuilder()
                .setDtype(DataType.DT_FLOAT) // Set the data type to float
                .setTensorShape(TensorShapeProto.newBuilder().addDim(TensorShapeProto.Dim.newBuilder().setSize(-1))) // Dynamic output size
                .setName("box_output") // Name of the box output tensor
                .build();
        signatureDefBuilder.putOutputs("box_output", boxOutputTensorInfo); // Add to signature

        // Attach the SignatureDef to the MetaGraphDef
        metaGraphDefBuilder.putSignatureDef("serving_default", signatureDefBuilder.build());

        // Create a SavedModel builder and add the MetaGraphDef to it
        SavedModel.Builder builder = SavedModel.newBuilder();
        builder.addMetaGraphs(metaGraphDefBuilder); // Add the MetaGraphDef to the SavedModel

        // Save the session's variables to the specified directory
        session.save(exportDir + "/model/variables/variables");

        // Write the MetaGraphDef to the saved_model.pb file within the model directory
        Files.write(modelDir.resolve("saved_model.pb"), builder.build().toByteArray());

        // Log the success message indicating where the model has been saved
        System.out.println("Model saved to " + modelDir);
    }

    // Method to update the loss values for box, class, and total losses, and refresh the graph
    public void updateLossValues(float box_l, float class_l, float total_l) {
        // Add the new box loss value for the current epoch
        boxLossValues.add(box_l);

        // Add the new class loss value for the current epoch
        classLossValues.add(class_l);

        // Add the new total loss value for the current epoch
        totalLossValues.add(total_l);

        // Repaint the graph to reflect the updated loss values
        repaint();
    }

    // Custom JPanel class for drawing the box loss graph
    static class box_loss_graph extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            // Use Graphics2D for advanced control over rendering
            Graphics2D g2d = (Graphics2D) g;

            // Enable antialiasing for smoother graphics
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Get the dimensions of the panel
            int panelWidth = getWidth();
            int panelHeight = getHeight();

            // Set margins for axis positioning
            int marginLeft = 30;
            int marginBottom = 30;

            // Define the origin point for the graph (bottom-left corner)
            int originY = panelHeight - marginBottom;

            // Draw the x-axis (epochs)
            g2d.drawLine(marginLeft, originY, panelWidth - marginLeft, originY);

            // Draw the y-axis (loss values)
            g2d.drawLine(marginLeft, originY, marginLeft, marginBottom);

            // Get the number of epochs to display
            int maxEpochs = boxLossValues.size();

            // Draw the loss values as a line graph across epochs
            for (int epoch = 1; epoch < maxEpochs; epoch++) {
                // Get the previous and current loss values for each epoch
                float lossPrev = boxLossValues.get(epoch - 1);
                float lossCurrent = boxLossValues.get(epoch);

                // Calculate the x and y coordinates, scaling for better visibility
                int x1 = marginLeft + (epoch - 1) * 2; // Scale x-axis (e.g., 2 pixels per epoch)
                int y1 = originY - (int) (lossPrev / maxLoss * (panelHeight - marginBottom - 30));  // Scale loss values
                int x2 = marginLeft + epoch * 2;  // Next epoch's x-coordinate
                int y2 = originY - (int) (lossCurrent / maxLoss * (panelHeight - marginBottom - 30));  // Next loss value

                // Draw the line connecting the points (previous and current epoch)
                g2d.drawLine(x1, y1, x2, y2);
            }
        }

        @Override
        public Dimension getPreferredSize() {
            // Return preferred size for the panel based on the number of epochs
            return new Dimension(epochs * 2, 230);
        }
    }

    // Custom JPanel class for drawing the class loss graph
    static class class_loss_graph extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            // Use Graphics2D for improved rendering control
            Graphics2D g2d = (Graphics2D) g;

            // Enable antialiasing for smoother edges
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Retrieve the dimensions of the panel
            int panelWidth = getWidth();
            int panelHeight = getHeight();

            // Set margins for the axes
            int marginLeft = 30;
            int marginBottom = 30;

            // Define the origin point for the graph (bottom-left corner)
            int originY = panelHeight - marginBottom;

            // Draw the x-axis representing epochs
            g2d.drawLine(marginLeft, originY, panelWidth - marginLeft, originY);

            // Draw the y-axis representing loss values
            g2d.drawLine(marginLeft, originY, marginLeft, marginBottom);

            // Determine the number of epochs to display
            int maxEpochs = classLossValues.size();

            // Plot the class loss values as a line graph
            for (int epoch = 1; epoch < maxEpochs; epoch++) {
                // Retrieve the previous and current loss values
                float lossPrev = classLossValues.get(epoch - 1);
                float lossCurrent = classLossValues.get(epoch);

                // Scale the x and y values for accurate drawing
                int x1 = marginLeft + (epoch - 1) * 2; // Scale x-axis (e.g., 2 pixels per epoch)
                int y1 = originY - (int) (lossPrev / maxLoss * (panelHeight - marginBottom - 30));  // Scale loss value
                int x2 = marginLeft + epoch * 2;  // Calculate x-coordinate for the next epoch
                int y2 = originY - (int) (lossCurrent / maxLoss * (panelHeight - marginBottom - 30));  // Scale loss value

                // Draw a line connecting the previous and current loss points
                g2d.drawLine(x1, y1, x2, y2);
            }
        }

        @Override
        public Dimension getPreferredSize() {
            // Return the preferred size of the graph panel based on the number of epochs
            return new Dimension(epochs * 2, 230);
        }
    }

    // Custom JPanel class for drawing the total loss graph
    static class total_loss_graph extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            // Use Graphics2D for enhanced rendering control
            Graphics2D g2d = (Graphics2D) g;

            // Enable antialiasing for smoother graphics
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Retrieve the dimensions of the panel
            int panelWidth = getWidth();
            int panelHeight = getHeight();

            // Define margins for the axes
            int marginLeft = 30;
            int marginBottom = 30;

            // Set the origin point for the graph (bottom-left corner)
            int originY = panelHeight - marginBottom;

            // Draw the x-axis representing epochs
            g2d.drawLine(marginLeft, originY, panelWidth - marginLeft, originY);

            // Draw the y-axis representing loss values
            g2d.drawLine(marginLeft, originY, marginLeft, marginBottom);

            // Determine the number of epochs to display
            int maxEpochs = totalLossValues.size();

            // Plot the total loss values as a line graph
            for (int epoch = 1; epoch < maxEpochs; epoch++) {
                // Retrieve the previous and current loss values
                float lossPrev = totalLossValues.get(epoch - 1);
                float lossCurrent = totalLossValues.get(epoch);

                // Scale the x and y values for accurate drawing
                int x1 = marginLeft + (epoch - 1) * 2; // Scale x-axis (e.g., 2 pixels per epoch)
                int y1 = originY - (int) (lossPrev / maxLoss * (panelHeight - marginBottom - 30));  // Scale loss value
                int x2 = marginLeft + epoch * 2;  // Calculate x-coordinate for the next epoch
                int y2 = originY - (int) (lossCurrent / maxLoss * (panelHeight - marginBottom - 30));  // Scale loss value

                // Draw a line connecting the previous and current loss points
                g2d.drawLine(x1, y1, x2, y2);
            }
        }

        @Override
        public Dimension getPreferredSize() {
            // Return the preferred size of the graph panel based on the number of epochs
            return new Dimension(epochs * 2, 230);
        }
    }
}