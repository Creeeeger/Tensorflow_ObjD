package org.tensorAction;

import org.object_d.CNNPlayground;
import org.object_d.Main_UI;
import org.tensorflow.*;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
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
import java.util.stream.IntStream;

public class tensorTrainerCNN extends JFrame {
    static int numberClasses, epochs;                           // Declare variables for the number of classes and epochs (training iterations)
    static float maxLoss = Main_UI.learning_rate;               // Initialize maxLoss with the learning rate value from the Main_UI class
    static List<Float> classLossValues = new ArrayList<>();     // List to store loss values for class predictions
    static List<Float> totalLossValues = new ArrayList<>();     // List to store total loss values
    static JTextArea textArea;                                  // Declare a text area for displaying output as well as the matrix
    static JLabel accuracy_label;                               // Label to display the accuracy of the model

    public tensorTrainerCNN() {
        setLayout(new GridLayout(4, 1, 10, 10)); // Set layout for the panel, a 4-row grid with spacing between elements

        // Initialize and configure the class loss panel
        JPanel class_loss_panel = new JPanel();
        class_loss_panel.setBorder(BorderFactory.createTitledBorder("Class loss")); // Set a titled border for the panel
        class_loss_graph classLossGraph = new class_loss_graph(); // Create an instance of the graph for class loss
        class_loss_panel.add(classLossGraph); // Add the graph to the panel

        // Initialize and configure the total loss panel
        JPanel total_loss_panel = new JPanel();
        total_loss_panel.setBorder(BorderFactory.createTitledBorder("Total loss")); // Set a titled border for the panel
        total_loss_graph totalLossGraph = new total_loss_graph(); // Create an instance of the graph for total loss
        total_loss_panel.add(totalLossGraph); // Add the graph to the panel

        // Initialize and configure the confusion matrix panel
        JPanel confusion_matrix_panel = new JPanel();
        confusion_matrix_panel.setBorder(BorderFactory.createTitledBorder("Confusion matrix")); // Set a titled border

        // Add the panels to the layout
        add(class_loss_panel);
        add(total_loss_panel);
        add(confusion_matrix_panel);

        // Initialize the text area for the confusion matrix and the accuracy label
        // Set text area size dynamically based on the number of classes
        // + 2 because the number row is one and the counting process requires one more
        textArea = new JTextArea(numberClasses + 2, numberClasses + 2);
        textArea.setEditable(false);
        accuracy_label = new JLabel("Final accuracy: ..."); // Label to display final accuracy

        // Add the accuracy label and text area to the confusion matrix panel
        confusion_matrix_panel.add(accuracy_label);
        confusion_matrix_panel.add(textArea);
    }

    /**
     * Accesses and trains a model using a specified folder containing image data
     * The folder should contain subdirectories representing different classes, each containing image files.
     *
     * @param folder   The path to the folder containing the image data organized into class subdirectories.
     * @param variable A boolean flag to enable or disable variable layer training.
     * @param layers   An ArrayList of layerEntry objects representing the layers used in the CNN (if variable training is enabled).
     * @throws IOException If there is an error while loading the dataset or processing the images.
     */
    public static void access(String folder, Boolean variable, ArrayList<CNNPlayground.layerEntry> layers) throws IOException {
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

        if (batchSize < 1) {
            throw new RuntimeException("Must have a greater batch size than 1");
        }

        // Check if the folder contains no subdirectories (i.e., no grouped images)
        if (numberClasses == 0) {
            throw new RuntimeException("You can't use a folder without grouped images!"); // Throw an error if no classes found
        }

        // Load the dataset into batches with specified parameters
        TFloat32[] datasetBatch = loadDataset(folder, batchSize, imageSize, imageSize, 3, numberClasses);
        TFloat32 images = datasetBatch[0]; // Extract the image batch
        TFloat32 labels = datasetBatch[1]; // Extract the corresponding labels batch

        // Train the model with the loaded dataset, number of classes, epochs, and image size
        // create switch for variable layer training and normal training
        if (variable) {
            trainModel(images, labels, numberClasses, tensorTrainerCNN.epochs, imageSize, true, layers);
        } else {
            trainModel(images, labels, numberClasses, tensorTrainerCNN.epochs, imageSize, false, null);
        }
    }

    /**
     * Loads and preprocesses the dataset from the specified directory, generating batches of images and their corresponding labels.
     *
     * @param dataDir     The path to the directory containing class subdirectories with image files.
     * @param batchSize   The number of images
     * @param imageHeight The desired height of the images after resizing.
     * @param imageWidth  The desired width of the images after resizing.
     * @param numChannels The number of color channels in the images (e.g., 3 for RGB images).
     * @param numClasses  The total number of classes
     * @return An array containing two TFloat32 tensors: one for the images and one for the labels.
     * @throws IOException If there is an error while loading the images or creating the tensors.
     */
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

    /**
     * Preprocesses an image by resizing it to the target dimensions and converting it to a normalized float array.
     * The image is resized to the specified height and width, and each pixel's RGB value is normalized to the range [0, 1].
     *
     * @param img          The input BufferedImage to be processed.
     * @param targetHeight The target height of the resized image.
     * @param targetWidth  The target width of the resized image.
     * @return A 3D float array representing the resized and normalized image, with dimensions [height][width][channels].
     */
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

    /**
     * Preprocesses a label for a given class label by converting it into a one-hot encoded array.
     *
     * @param classLabel The integer representing the class label to be encoded.
     * @param numClasses The total number of classes
     * @return A one-hot encoded label array of length `numClasses`, with 1.0 at the position of the class label and 0.0 elsewhere.
     */
    public static float[] preprocessLabel(int classLabel, int numClasses) {
        // Create an array to hold the one-hot encoded label, initialized to zero
        float[] labelArray = new float[numClasses];

        // Set the position corresponding to the class label to 1.0 for one-hot encoding
        labelArray[classLabel] = 1.0f;  // One-hot encoding

        // Return the one-hot encoded label array
        return labelArray;
    }

    /**
     * Resizes a given image to the specified dimensions (height and width).
     *
     * @param img          The input BufferedImage to be resized.
     * @param targetHeight The target height of the resized image.
     * @param targetWidth  The target width of the resized image.
     * @return A new BufferedImage that represents the resized image.
     */
    public static BufferedImage resizeImage(BufferedImage img, int targetHeight, int targetWidth) {
        // Create a new BufferedImage with the target dimensions and RGB color model
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);

        // Draw the original image onto the new resized image with the specified dimensions
        resizedImage.getGraphics().drawImage(img, 0, 0, targetWidth, targetHeight, null);

        // Return the resized image
        return resizedImage;
    }

    /**
     * Constructs a computational graph for building and training a Convolutional Neural Network
     *
     * @param numClasses The number of output classes for classification
     * @param imageSize  The height and width of the input images
     * @param layers     list of layers - null if no dynamic layers otherwise list contains layer configurations
     * @return A TensorFlow computational graph
     * @throws IllegalArgumentException If the `layers` list contains unsupported layer types or invalid parameters.
     */
    public static Graph Graph(int numClasses, int imageSize, ArrayList<CNNPlayground.layerEntry> layers) {
        // Define constants for the number of channels and random seed for initialization
        final int NUM_CHANNELS = 3; // RGB image
        final boolean variable = (layers == null); // check if layers is null since if so we don't use layers

        // Create a new computation graph
        Graph graph = new Graph();
        Ops tf = Ops.create(graph); // TensorFlow operations instance

        // Input placeholders
        // Input tensor for image data, shape: [batch_size, imageSize, imageSize, NUM_CHANNELS]
        Placeholder<TFloat32> input = tf.withName("input").placeholder(
                TFloat32.class,
                Placeholder.shape(
                        Shape.of(
                                -1,
                                imageSize,
                                imageSize,
                                NUM_CHANNELS
                        )
                )
        );

        // Reshape input tensor if necessary
        Reshape<TFloat32> inputReshaped = tf.reshape(
                input,
                tf.array(
                        -1,
                        imageSize,
                        imageSize,
                        NUM_CHANNELS
                )
        );

        // Placeholder for class labels
        Placeholder<TFloat32> classLabels = tf.withName("labels").placeholder(
                TFloat32.class,
                Placeholder.shape(
                        Shape.of(
                                -1,
                                numClasses
                        )
                )
        ); // shape for class labels

        // Input normalization (feature scaling)
        // Scale pixel values from [0, 255] to [-0.5, 0.5] for better training convergence
        Operand<TFloat32> scaledInput = tf.math.div(
                tf.math.sub(
                        tf.dtypes.cast(
                                inputReshaped,
                                TFloat32.class
                        ),
                        tf.constant(127.5f)),
                tf.constant(255.0f)
        );
        /*
        Casting Input to TFloat32 and subtracting 127.5 so:
         tf.math.sub(
             tf.dtypes.cast(inputReshaped, TFloat32.class),
             tf.constant(127.5f)
         )
        •	Original range: [0, 255].
	    •	After subtraction: [-127.5, 127.5].

	    Dividing by a Constant tf.math.div(..., tf.constant(255.0f))
	    •	range [-127.5, 127.5] to [-0.5, 0.5].
        •	Minimum value: -127.5 / 255.0 = -0.5.
        •	Maximum value: 127.5 / 255.0 = 0.5.
         */

        if (variable) { // no dynamic layers
            // Build Convolutional Layers followed by Max Pooling layers
            Operand<TFloat32> conv1 = buildConvLayer(tf, scaledInput, NUM_CHANNELS, 32); // First convolution layer
            Operand<TFloat32> pool1 = buildMaxPoolLayer(tf, conv1); // First max pooling layer

            Operand<TFloat32> conv2 = buildConvLayer(tf, pool1, 32, 64); // Second convolution layer
            Operand<TFloat32> pool2 = buildMaxPoolLayer(tf, conv2); // Second max pooling layer

            Operand<TFloat32> conv3 = buildConvLayer(tf, pool2, 64, 128); // Third convolution layer
            Operand<TFloat32> pool3 = buildMaxPoolLayer(tf, conv3); // Third max pooling layer

            // Flatten the output from the last pooling layer to feed into fully connected layers
            Operand<TFloat32> flatten = tf.reshape(pool3,
                    tf.concat(
                            Arrays.asList(
                                    tf.slice(tf.shape(pool3), tf.array(0), tf.array(1)), // Keep the batch size
                                    tf.array(-1)
                            ),
                            tf.constant(0)
                    )
            ); // Flatten other dimensions
            /*
             tf.slice(tf.shape(pool3), tf.array(0), tf.array(1))
                •	tf.shape(pool3): data (shape of tensor)
	            •	tf.array(0): Starting index of the slice (0th index = batch size).
	            •	tf.array(1): Number of elements to slice (just the batch size).
	            so return the batch size

	         tf.array(-1)
	            •	dimension size for the flattened portion (-1 to keep it dynamic)
	            size will be height*width*channels

	         Arrays.asList(...)
	            combine batch size and dynamic flatten part into a list

	         tf.concat(..., tf.constant(0))
	            •	Concatenates the batch size and flattened dimension into a single tensor along a specified axis.
            	•	tf.constant(0): Specifies axis 0 (concatenating along the same dimension).

             tf.reshape(pool3, ...)
                •	Reshapes the tensor pool3 into new shape [batch_size, -1].
                •	Transforms the 4D tensor into a 2D tensor:
             */

            int inputs = (int) (pool3.shape().get(1) * pool3.shape().get(2) * pool3.shape().get(3)); //fix amount of inputs for avoiding errors and improving performance

            // Fully Connected Layers
            Operand<TFloat32> fc1 = buildFullyConnectedLayer(tf, flatten, inputs, 512); // First fully connected layer

            // Classification Output using Softmax activation
            Operand<TFloat32> logits = buildFullyConnectedLayer(tf, fc1, 512, numClasses); // Fully connected layer for class logits
            tf.withName("class_output").nn.softmax(logits); // Apply softmax to logits for class probabilities

            // Compute softmax cross-entropy loss for classification
            SoftmaxCrossEntropyWithLogits<TFloat32> crossEntropy = tf.nn.softmaxCrossEntropyWithLogits(logits, classLabels);
            Mean<TFloat32> classLoss = tf.math.mean(crossEntropy.loss(), tf.constant(0)); // Mean cross-entropy loss

            // Regularization (L2 Loss) to prevent overfitting
            Add<TFloat32> regularizers = tf.math.add(
                    tf.nn.l2Loss(fc1),
                    tf.nn.l2Loss(logits)
            ); // L2 loss for fully connected layer and logits

            // Compute total loss as the sum of class loss and regularization
            Add<TFloat32> totalLoss = tf.withName("totalLoss").math.add(
                    classLoss,
                    tf.math.mul(
                            regularizers,
                            tf.constant(5e-4f)
                    )
            ); // Scale regularization term

            // Optimizer (Adam) for minimizing the total loss
            Optimizer optimizer = new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-6f); // Create an Adam optimizer
            optimizer.minimize(totalLoss, "train"); // Add minimization operation to the optimizer
        } else { // Dynamic layers
            // Initialize the current layer with the scaled input image
            Operand<TFloat32> currentLayer = scaledInput;

            // Set the number of input channels to 3 (RGB) and the number of output channels for the first convolution layer to 32
            int localChannelsIn = NUM_CHANNELS;
            int channelsOut = 32;

            // Iterate over the layers in the provided list and dynamically add them to the graph
            for (CNNPlayground.layerEntry layer : layers) {
                String layerType = layer.getLayer();  // Get the type of the current layer
                ArrayList<Double> params = layer.getList();  // Get the parameters for the current layer

                switch (layerType) {
                    case "Convolutional Layer":
                        // Extract the kernel size, stride, weight scaling factor, and seed from the layer's parameters
                        double kernelSize = params.get(0);
                        double stride = params.get(1);
                        double weightScale = params.get(2);
                        double seed = params.get(3);

                        // Build the convolutional layer and update the current layer
                        // The number of output channels will increase after each convolution layer
                        currentLayer = buildVariableConvLayer(tf, currentLayer, (int) kernelSize, localChannelsIn, channelsOut, (long) stride, (float) weightScale, (long) seed);

                        // Update the input and output channels for the next layer
                        localChannelsIn = channelsOut;  // The input channels for the next layer is the output channels of the current layer
                        channelsOut *= 2;               // Double the output channels for the next convolution layer
                        break;

                    case "Pooling Layer":
                        // Extract the kernel size and padding parameter for the pooling layer
                        double kernelSizeP = params.get(0);

                        // Validate the padding parameter (0 for VALID, 1 for SAME)
                        if (params.get(1) == null || (params.get(1) != 0 && params.get(1) != 1)) {
                            throw new IllegalArgumentException("Invalid padding parameter for pooling layer.");
                        }

                        // Convert the padding value to a string ("VALID" or "SAME")
                        String paddingConverted = (params.get(1) == 0) ? "VALID" : "SAME";

                        // Build the max pooling layer and update the current layer
                        currentLayer = buildVariableMaxPoolLayer(tf, currentLayer, (int) kernelSizeP, paddingConverted);
                        break;

                    case "Fully Connected Layer":
                        // Extract parameters for the fully connected layer: weight initialization scale, bias initialization scale, and seed
                        double weightInitScale = params.get(0);
                        double biasInitScale = params.get(1);
                        double seedFC = params.get(2);

                        // Get the height, width, and channels of the current layer's output
                        int height = (int) currentLayer.shape().get(1);  // Height of the feature map
                        int width = (int) currentLayer.shape().get(2);   // Width of the feature map
                        int channels = (int) currentLayer.shape().get(3); // Number of channels
                        int units = height * width * channels;           // Flattened size of the feature map (total units)

                        // Flatten the output of the previous layer to prepare for the fully connected layer
                        currentLayer = tf.reshape(currentLayer, tf.concat(Arrays.asList(
                                tf.slice(tf.shape(currentLayer), tf.array(0), tf.array(1)), // Keep batch size
                                tf.array(-1)), tf.constant(0))); // Flatten the remaining dimensions

                        // Build the fully connected layer and update the current layer
                        currentLayer = buildVariableFullyConnectedLayer(tf, currentLayer, units, units, (float) weightInitScale, (float) biasInitScale, (long) seedFC);

                        // Un-flatten the output back into a 4D tensor after the fully connected layer
                        currentLayer = tf.reshape(currentLayer, tf.concat(Arrays.asList(
                                tf.slice(tf.shape(currentLayer), tf.array(0), tf.array(1)), // Keep batch size
                                tf.array(height, width, channels) // Un-flatten to the original dimensions
                        ), tf.constant(0)));
                        /*
                        extract the batch size      tf.slice(tf.shape(currentLayer), tf.array(0), tf.array(1)), // Keep batch size
                        create new array with previously saved dimensions   tf.array(height, width, channels) // Un-flatten to the original dimensions
                        put them together as a list     Arrays.asList(
                        concat them along the same axis tf.concat(..., tf.constant(0))
                        reshape data
                         */
                        break;

                    default:
                        // If the layer type is not supported, throw exception
                        throw new IllegalArgumentException("Unsupported layer type: " + layerType);
                }
            }

            // Flatten the tensor before feeding it into the classification output layer
            int inputs = (int) (currentLayer.shape().get(1) * currentLayer.shape().get(2) * currentLayer.shape().get(3));
            currentLayer = tf.reshape(currentLayer, tf.concat(Arrays.asList(
                    tf.slice(tf.shape(currentLayer), tf.array(0), tf.array(1)), // Keep batch size
                    tf.array(-1)), tf.constant(0))); // Flatten the remaining dimensions

            // Build the final fully connected layer that outputs logits (un-normalized class scores)
            Operand<TFloat32> logits = buildFullyConnectedLayer(tf, currentLayer, inputs, numClasses);

            // Apply the softmax function to the logits to convert them into class probabilities
            tf.withName("class_output").nn.softmax(logits);

            // Compute the softmax cross-entropy loss for the classification task
            Mean<TFloat32> classLoss = tf.math.mean(
                    tf.nn.softmaxCrossEntropyWithLogits(
                            logits,
                            classLabels
                    ).loss(),
                    tf.constant(0)
            );

            // Regularization (L2 Loss) to prevent overfitting by penalizing large weights
            Add<TFloat32> regularizers = tf.math.add(tf.nn.l2Loss(currentLayer), tf.nn.l2Loss(logits));

            // Compute the total loss as the sum of class loss and regularization term
            Add<TFloat32> totalLoss = tf.withName("totalLoss").math.add(classLoss, tf.math.mul(regularizers, tf.constant(5e-4f)));

            // Create an Adam optimizer to minimize the total loss during training
            Optimizer optimizer = new Adam(graph, 0.005f, 0.9f, 0.999f, 1e-6f); // Adam optimizer with specified parameters
            optimizer.minimize(totalLoss, "train"); // Add the minimization operation to the optimizer
        }

        return graph; // Return the constructed computation graph
    }

    /**
     * Builds a Convolutional Layer followed by a ReLU activation.
     *
     * @param tf             The TensorFlow operations (Ops) object, used to create and manipulate tensors.
     * @param input          The input tensor to the convolutional layer, representing an image or feature map.
     * @param inputChannels  The number of channels in the input tensor (e.g., 3 for RGB).
     * @param outputChannels The number of output channels (i.e., number of filters for the convolution).
     * @return The result of applying the convolutional operation, bias addition, and ReLU activation.
     * @throws IllegalArgumentException If inputChannels or outputChannels are non-positive.
     */
    private static Operand<TFloat32> buildConvLayer(Ops tf, Operand<TFloat32> input, int inputChannels, int outputChannels) {
        // Create the filter of a 5x5 kernel
        // inputChannels -> depth
        // outputChannels -> number of filters
        // Truncated normal distribution is used -> limiting extreme values
        Operand<TFloat32> convWeights = tf.variable(
                tf.math.mul(
                        tf.random.truncatedNormal(
                                tf.array( // Shape of the filter
                                        5,
                                        5,
                                        inputChannels,
                                        outputChannels
                                ),
                                TFloat32.class, // Data type
                                TruncatedNormal.seed(12345L) // Fixed seed for reproducibility
                        ),
                        tf.constant(0.1f) // Constant to multiply weight by
                )
        ); // Scale the randomly initialized weights by 0.1
        /*
        Filter (Weights) Initialization:
            Weights decide how features are extracted from data
            Random Initialization of Weights:
            kernel array is filled with random normalized truncated values
            Shape of Weights: tf.array(5, 5, inputChannels, outputChannels)
            5, 5: The filter size (kernel is 5*5)
            inputChannels: The depth of the input feature map (so for an RGB image we have 3 Channels so a depth of 3. so each filter has 3 layers for each channel)
            outputChannels: Number of Filters and input for the next layer

            Truncated Normal Distribution:
            tf.random.truncatedNormal(...)
            Generates random numbers which form a normal distribution, values are truncated to prevent extremes values
            seed(12345L): Ensures reproducibility. same input - same output

            Scaling the Weights:
            tf.math.mul(..., tf.constant(0.1f))
            Multiplies each weight by 0.1
            prevent large gradients, keep weights small -> better training

            Defining the Variable:
            tf.variable(...)
            forms data into a trainable variable
         */

        // Create a bias for the operation, initialized to 0 for each output channel
        Operand<TFloat32> convBiases = tf.variable(tf.fill(tf.array(outputChannels), tf.constant(0.0f)));
        /*
         Bias Initialization
         create an array with as many slots as output channels and set each slot (bias) to 0.0
         so there are no value shifts
         */

        // Perform the 2D convolution operation
        // "SAME" padding ensures that the output has the same dimensions as the input
        Conv2d<TFloat32> conv = tf.nn.conv2d(
                input,
                convWeights,
                Arrays.asList(1L, 1L, 1L, 1L),
                "SAME"
        );
        /*
        Convolution Operation:
        Applies filters (convWeights) to input tensor using a 2D convolution operation.
	    Input: input 4D tensor, shape [batchSize, height, width, Channels].
	    Filters: convWeights are filters for extracting features
	    Strides: Arrays.asList(1L, 1L, 1L, 1L) shifting of filter (no skipping of pixels, moves 1 pixel a time)
	    Padding: "SAME" -> output tensor has same dimensions as the input one;
         */

        // Add the biases to the convolution result, shifting the values for each output channel
        BiasAdd<TFloat32> biasAdd = tf.nn.biasAdd(
                conv,
                convBiases
        );
        /*
        Bias Addition:
        biasAdd = conv + convBiases
        add the bias to the convolutional operations
         */

        // Apply the ReLU activation function to introduce non-linearity. ReLU sets negative values to 0 and keeps positive values as they are
        return tf.nn.relu(biasAdd); // -> learn more complex patterns
        /*
         Activation Function (ReLU):
         f(x) = max(0, x)
         negative values to 0
         keep positive ones the same
         */
    }

    /**
     * Builds a Max Pooling Layer for down sampling the input tensor.
     *
     * @param tf    The TensorFlow operations (Ops) object, used to create and manipulate tensors.
     * @param input The input tensor to the max pooling layer, representing an image or feature map.
     * @return The result of applying the max pooling operation.
     * @throws IllegalArgumentException If the input tensor is not compatible with max pooling.
     */
    private static Operand<TFloat32> buildMaxPoolLayer(Ops tf, Operand<TFloat32> input) {
        // Apply max pooling with a 2x2 filter which halves the height and width of the input
        // "SAME" padding ensures that the output size is reduced evenly
        return tf.nn.maxPool(
                input,
                tf.array(1, 2, 2, 1),
                tf.array(1, 2, 2, 1),
                "SAME"
        );
        /*
        input
        Filter Size:
        	•	Dimensions:
	            •	1: Batch dimension (no pooling across batches).
	            •	2: Height of the filter (2 pixels).
            	•	2: Width of the filter (2 pixels).
	            •	1: Channels (no pooling across channels).

	    Strides:
	    	•	Dimensions:
	            •	1: Batch dimension (no stride across batches).
	            •	2: Vertical stride (moves 2 pixels vertically).
	            •	2: Horizontal stride (moves 2 pixels horizontally).
	            •	1: Channels (no stride across channels).

	     Padding:
	     	•	Padding Mode:
	            •	"SAME" dimensions reduced evenly
         */

        // Max Pooling --> keep important features while reducing complexity and size
    }

    /**
     * Builds a Fully Connected Layer / Dense layer + ReLU activation.
     *
     * @param tf          The TensorFlow operations (Ops) object, used to create and manipulate tensors.
     * @param input       The input tensor to the fully connected layer, representing a flattened feature map.
     * @param inputUnits  The number of input units (features)
     * @param outputUnits The number of output units (neurons)
     * @return The result of applying matrix multiplication, bias addition, and ReLU activation.
     * @throws IllegalArgumentException If inputUnits or outputUnits are non-positive.
     */
    private static Operand<TFloat32> buildFullyConnectedLayer(Ops tf, Operand<TFloat32> input, int inputUnits, int outputUnits) {
        // Initialize the weights matrix for the fully connected layer with inputUnits (number of input features) and outputUnits (number of neurons)
        Operand<TFloat32> weights = tf.variable(
                tf.math.mul(
                        tf.random.truncatedNormal(
                                tf.array(inputUnits, outputUnits),
                                TFloat32.class,
                                TruncatedNormal.seed(12345L)
                        ),
                        tf.constant(0.1f)
                )
        );
        /*
	        •	Weights Matrix:
            tf.random.truncatedNormal(tf.array(inputUnits, outputUnits), TFloat32.class, TruncatedNormal.seed(12345L))
                tf.array(inputUnits, outputUnits):
            	•   shape of weight tensor
	            •	inputUnits: Number of input features (columns in the weight matrix).
            	•	outputUnits: Number of output neurons (rows in the weight matrix).

            	TFloat32.class:
	            •	data type -> 32-bit floating-point numbers

	            TruncatedNormal.seed(12345L):
	            •	reproducibility -> same random values will generate every time code runs

            tf.math.mul(..., tf.constant(0.1f)) - Multiplies all values by 0.1 to scale down

	        tf.variable(...) - create a trainable variable used later
         */

        // Initialize biases for each output unit (neuron), set to a small positive value (0.1) to avoid "dead neurons"
        Operand<TFloat32> biases = tf.variable(tf.fill(tf.array(outputUnits), tf.constant(0.1f)));
        /*
            Bias Vector:
        	    array with slots as many as output elements and every element has the value of 0.1
         */

        // Perform matrix multiplication between the input and weights, combines features across neurons
        // Add biases to the result of the matrix multiplication, shifting the values before applying activation
        Operand<TFloat32> dense = tf.math.add(tf.linalg.matMul(input, weights), biases);
        /*
            dense = (input * weights) + biases
	        •	Matrix Multiplication: input * weights across neurons.
	        •	Bias Addition: Shifts values for each neuron.
         */

        // Apply the ReLU activation function to the result of the fully connected layer to introduce non-linearity
        return tf.nn.relu(dense);
        /*
            f(x) = max(0, x)
	        •	Introduces non-linearity -> learn complex relationships
         */
    }

    /**
     * Build 2D convolutional layer with weights and biases.
     *
     * @param tf             TensorFlow Ops for building the graph.
     * @param input          The input tensor of shape [batch, height, width, inputChannels].
     * @param kernelSize     The size of the convolutional kernel (filter), e.g., 5 for a 5x5 kernel.
     * @param inputChannels  The number of input channels (depth of the input tensor).
     * @param outputChannels The number of output channels (number of filters).
     * @param stride         The stride of the convolution, typically 1 for no down sampling.
     * @param weightScale    The scale to initialize the weights, e.g., 0.1f to scale the weights.
     * @param seed           The random seed to ensure reproducibility for weight initialization.
     * @return The output tensor after applying the convolution, bias addition, and ReLU activation.
     */
    private static Operand<TFloat32> buildVariableConvLayer(Ops tf, Operand<TFloat32> input, int kernelSize, int inputChannels,
                                                            int outputChannels, long stride, float weightScale, long seed) {
        // Initialize the convolutional weights with a truncated normal distribution
        Operand<TFloat32> convWeights = tf.variable(tf.math.mul(
                tf.random.truncatedNormal(tf.array(kernelSize, kernelSize, inputChannels, outputChannels),
                        TFloat32.class, TruncatedNormal.seed(seed)),
                tf.constant(weightScale)));

        // Initialize the biases as zeros
        Operand<TFloat32> convBiases = tf.variable(tf.fill(tf.array(outputChannels), tf.constant(0.0f)));

        // Perform the 2D convolution operation
        Conv2d<TFloat32> conv = tf.nn.conv2d(input, convWeights, Arrays.asList(1L, stride, stride, 1L), "SAME");

        // Add the biases to the convolution result
        BiasAdd<TFloat32> biasAdd = tf.nn.biasAdd(conv, convBiases);

        // Apply the ReLU activation function
        return tf.nn.relu(biasAdd);
    }

    /**
     * Builds a Max Pooling layer with a configurable kernel size and padding.
     *
     * @param tf         TensorFlow Ops object for building the computational graph.
     * @param input      The input tensor to the Max Pooling layer (shape: [batchSize, height, width, channels]).
     * @param kernelSize The size of the pooling window (applied equally to height and width).
     * @param padding    The padding algorithm to use, either "SAME" or "VALID".
     * @return output of Max Pooling operation.
     */
    private static Operand<TFloat32> buildVariableMaxPoolLayer(Ops tf, Operand<TFloat32> input, int kernelSize, String padding) {
        return tf.nn.maxPool(input, tf.array(1, kernelSize, kernelSize, 1), tf.array(1, kernelSize, kernelSize, 1), padding);
    }

    /**
     * Build fully connected layer for neural network.
     *
     * @param tf              TensorFlow Ops object for building graph.
     * @param input           The input tensor to the fully connected layer (shape: [batchSize, inputUnits]).
     * @param inputUnits      Number of input features (units).
     * @param OutputUnits     Number of output features (neurons in the layer).
     * @param weightInitScale Scale factor for initializing weights.
     * @param biasInitValue   Initial value for biases (e.g., 0.1 to avoid "dead neurons").
     * @param seed            Seed value for random initialization of weights.
     * @return output of fully connected layer.
     */
    private static Operand<TFloat32> buildVariableFullyConnectedLayer(Ops tf, Operand<TFloat32> input, int inputUnits, int OutputUnits,
                                                                      float weightInitScale, float biasInitValue, long seed) {
        // Initialize the weights matrix for the fully connected layer
        Operand<TFloat32> weights = tf.variable(tf.math.mul(
                tf.random.truncatedNormal(tf.array(inputUnits, OutputUnits), TFloat32.class, TruncatedNormal.seed(seed)),
                tf.constant(weightInitScale)
        ));

        // Initialize biases for each output unit
        Operand<TFloat32> biases = tf.variable(tf.fill(tf.array(OutputUnits), tf.constant(biasInitValue)));

        // Perform matrix multiplication between the input and weights and add the biases
        Operand<TFloat32> dense = tf.math.add(tf.linalg.matMul(input, weights), biases);

        // Apply the ReLU activation function
        return tf.nn.relu(dense);
    }

    /**
     * Trains a CNN model using the provided images and labels for the specified number of epochs.
     *
     * @param images     The input images as a tensor (of type TFloat32), used for training the model.
     * @param labels     The correct labels as a tensor (of type TFloat32), used for supervised training.
     * @param numClasses The number of output classes for the classification task.
     * @param epochs     The number of epochs to train the model for.
     * @param imageSize  The size of the images
     * @param variable   Boolean flag for activation of variable layers
     * @param layers     A list of layer configurations (layerEntry objects), defining the layers of the CNN when variable layers are activated
     * @throws RuntimeException If an error occurs during model training or saving.
     */
    public static void trainModel(TFloat32 images, TFloat32 labels, int numClasses, int epochs, int imageSize, Boolean variable, ArrayList<CNNPlayground.layerEntry> layers) {
        // Initialize and display the live training analysis GUI window
        tensorTrainerCNN gui = new tensorTrainerCNN();
        gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set the window to hide on close
        gui.setTitle("Live training analysis"); // Set the window title
        gui.setVisible(true); // Make the window visible
        gui.setLocation(100, 10); // Position the window on the screen
        gui.pack(); // Adjust the window to fit its content

        if (!(epochs > 0)) {
            throw new RuntimeException("Must have more than 1 epoch");
        }

        // Create a new computation graph and session
        try {
            Graph graph;
            // variable initializer for layers
            if (variable) {
                graph = Graph(numClasses, imageSize, layers);
            } else {
                graph = Graph(numClasses, imageSize, null);
            }

            Session session = new Session(graph);
            // Initialize the Adam optimizer
            new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-6f);

            Result outputs = null;

            // Loop over the specified number of training epochs
            for (int epoch = 0; epoch < epochs; epoch++) {
                // Run the session, feeding the images, labels and target training
                Session.Runner runner = session.runner()
                        .feed("input", images)   // Feed image data
                        .feed("labels", labels)  // Feed label data
                        .addTarget("train");     // Target the "train" operation for optimization

                // Fetch the loss values for different components (class loss, total loss)
                outputs = runner
                        .fetch("class_output")
                        .fetch("totalLoss")
                        .run();                           // Run the session and collect the results

                // Get the loss values from the fetched outputs
                TFloat32 classTensor = (TFloat32) outputs.get(0);  // Class loss
                TFloat32 totalTensor = (TFloat32) outputs.get(1);  // Total loss

                // Print the loss values for the current epoch
                System.out.printf("Loss at epoch %d: %-10.6f %-10.6f%n", epoch, classTensor.getFloat(), totalTensor.getFloat());

                // Update the GUI with the new loss values for live visualization
                gui.updateLossValues(classTensor.getFloat(), totalTensor.getFloat());

                // Close output tensors to free resources
                for (Map.Entry<String, Tensor> tensor : outputs) {
                    tensor.getValue().close();
                }
            }

            // Print completion message after training is done
            System.out.println("Training completed.");

            //GOD BLESS THE MODEL SAVER WORKS
            // Create a session and function
            SessionFunction function = new SessionFunction(
                    new Signature.Builder().build(),
                    session
            ).withNewSession(session);

            // Create the main model directory
            Path modelDir = Paths.get("").toAbsolutePath().getParent().resolve("model"); // Resolving the "model" directory path

            // Create the model directory if it doesn't exist
            Files.createDirectories(modelDir);

            // Save the graph with the new operations
            SavedModelBundle.exporter(modelDir.toString())
                    .withTags("serve")
                    .withSession(session)
                    .withFunction(function)
                    .export();

            // Validate the model using the labels and outputs after training
            validate(labels, numClasses, Objects.requireNonNull(outputs));

            // Print message confirming model save
            System.out.println("Model saved");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Validates the model by comparing its predictions against the true labels.
     * It calculates the model's accuracy and generates a confusion matrix
     *
     * @param label      The true labels for the batch of images
     * @param numClasses The number of possible output classes
     * @param outputs    The model's prediction outputs, which contain the softmax probabilities for each class.
     */
    public static void validate(TFloat32 label, int numClasses, Result outputs) {
        int correctCount = 0; // Variable to track the number of correct predictions
        int[][] confusionMatrix = new int[numClasses][numClasses]; // Initialize confusion matrix with size [numClasses x numClasses]
        TFloat32 classPredictionTensor = (TFloat32) outputs.get(0); // Retrieve the prediction tensor containing softmax probabilities

        // Extract the batch size (number of samples in the batch) and number of classes from the tensor shape
        int batchSize = (int) classPredictionTensor.shape().get(0); // Number of images in the batch
        int numClassesFromTensor = (int) classPredictionTensor.shape().get(1); // Number of classes (dimension of the softmax output)

        // Ensure the number of classes matches between the given parameter and the tensor's shape
        if (numClasses != numClassesFromTensor) {
            throw new IllegalArgumentException("Mismatch between numClasses and tensor dimensions.");
            // Throw an error if there's a mismatch, as it would cause invalid indexing
        }

        // Process predictions for each image in parallel using Java Streams
        // Map each image index to its predicted label based on the class with the highest probability
        int[] predictedLabels = IntStream.range(0, batchSize) // Create a stream of image indices from 0 to batchSize-1
                .parallel() // Enable parallel processing for faster computation on larger batches
                .map(i -> { // Map each index to its corresponding predicted label
                    float maxProb = 0; // Initialize a variable to track the maximum softmax probability
                    int predictedLabel = 0; // Initialize the predicted class label

                    // Iterate over all classes to find the one with the highest probability
                    for (int j = 0; j < numClasses; j++) {
                        float prob = classPredictionTensor.getFloat(i, j); // Retrieve the probability for class `j` for image `i`
                        if (prob > maxProb) { // Update max probability and predicted label if current probability is higher
                            maxProb = prob;
                            predictedLabel = j;
                        }
                    }
                    return predictedLabel; // Return the predicted label for the current image
                })
                .toArray(); // Collect all predicted labels into an array

        // Calculate accuracy and update the confusion matrix
        for (int i = 0; i < batchSize; i++) { // Loop through each image in the batch
            int trueLabel = argmaxLabel(label, i); // Retrieve the true label for image `i` using the argmax function
            System.out.println("Predicted label for image " + i + ": " + predictedLabels[i] + " True label: " + trueLabel); // Print predicted and true labels for debugging/logging

            // Check if the prediction matches the true label
            if (predictedLabels[i] == trueLabel) {
                correctCount++; // Increment the correct prediction count
            }

            // Update the confusion matrix
            // confusionMatrix[trueLabel][predictedLabel] represents the count of true vs. predicted occurrences
            synchronized (confusionMatrix) {
                // Synchronize to ensure thread safety as the confusion matrix is shared across threads
                confusionMatrix[trueLabel][predictedLabels[i]]++;
            }
        }

        // Calculate the overall accuracy as the ratio of correct predictions to total images
        float accuracy = (float) correctCount / batchSize;
        System.out.println("Final accuracy: " + accuracy); // Print the calculated accuracy

        // Generate a string representation of the confusion matrix for display purposes
        StringBuilder matrixString = getStringBuilder(confusionMatrix);
        System.out.println(matrixString); // Print the confusion matrix to the console

        // Update the GUI components to display results
        textArea.setText(matrixString.toString()); // Display the confusion matrix in the text area
        accuracy_label.setText("Final accuracy: " + accuracy); // Display the final accuracy in the accuracy label
    }

    /**
     * Retrieves the index of the class with the maximum value from the tensor for a given iteration.
     *
     * @param tensor    The tensor containing the one-hot encoded labels, of type TFloat32.
     * @param iteration The index (iteration) for which the maximum value is to be found.
     * @return The index of the class with the maximum value (1.0 in a one-hot encoded label).
     */
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

    /**
     * Builds and formats the confusion matrix into a string representation for display.
     *
     * @param confusionMatrix The confusion matrix as a 2D array of integers, representing true vs. predicted labels.
     * @return A StringBuilder containing the formatted string representation of the confusion matrix.
     */
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

    /**
     * Updates the loss values for class loss and total loss, and refreshes the graph to visualize the training progress.
     *
     * @param class_l The class loss value for the current epoch.
     * @param total_l The total loss value for the current epoch.
     */
    public void updateLossValues(float class_l, float total_l) {
        // Add the new class loss value for the current epoch
        classLossValues.add(class_l);

        // Add the new total loss value for the current epoch
        totalLossValues.add(total_l);

        // Repaint the graph to reflect the updated loss values
        repaint();
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