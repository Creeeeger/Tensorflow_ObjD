package org.tensorAction;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.proto.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

import javax.imageio.IIOException;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


public class tensorTrainer {
    private static final float LEARNING_RATE = 0.1f;
    static File[] classDirs;

    public static void main(String[] args) throws IOException {
        String dataDir = "/Users/gregor/Desktop/Tensorflow_ObjD/flower_photos";
        List<Tensor> imageTensors = new ArrayList<>();
        List<Tensor> labelTensors = new ArrayList<>();

        classDirs = new File(dataDir).listFiles(File::isDirectory);
        Tensor imageTensor;
        Tensor labelTensor;
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
                                imageTensor = preprocessImage(img);
                                labelTensor = preprocessLabel(classLabel, classDirs.length);
                                imageTensors.add(imageTensor);
                                labelTensors.add(labelTensor);
                            }
                        } catch (IIOException e) {
                            // Print the error message and skip processing this image
                            System.out.println("Error reading image file: " + imageFile.getName() + " - " + e.getMessage());
                            continue; // Skip to the next iteration of the loop
                        } catch (IOException e) {
                            // Handle other IOExceptions if necessary
                            System.out.println("IOException occurred: " + e.getMessage());
                            continue; // Skip to the next iteration of the loop
                        }
                    }
                }
            }
        }
        System.out.println("Dataset loaded. Number of photos: " + imageTensors.size());

        // Convert imageTensors list to ByteNdArray
        ByteNdArray imageNdArray = NdArrays.ofBytes(Shape.of(imageTensors.size(), 224, 224, 3));
        for (int i = 0; i < imageTensors.size(); i++) {
            imageNdArray.slice(Indices.at(i)).copyTo(imageTensors.get(i).asRawTensor().data());
        }

        // Convert labelTensors list to ByteNdArray
        ByteNdArray labelNdArray = NdArrays.ofBytes(Shape.of(labelTensors.size()));
        for (int i = 0; i < labelTensors.size(); i++) {
            labelNdArray.slice(Indices.at(i)).copyTo(labelTensors.get(i).asRawTensor().data());
        }

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Create placeholders and variables, which should fit batches of an unknown number of images
            Placeholder<TFloat32> images = tf.placeholder(TFloat32.class);
            Placeholder<TFloat32> labels = tf.placeholder(TFloat32.class);

            // Create weights with an initial value of 0
            Shape weightShape = Shape.of(imageNdArray.get(0).shape().size(), classDirs.length);
            Variable<TFloat32> weights = tf.variable(tf.zeros(tf.constant(weightShape), TFloat32.class));

            // Create biases with an initial value of 0
            Shape biasShape = Shape.of(classDirs.length);
            Variable<TFloat32> biases = tf.variable(tf.zeros(tf.constant(biasShape), TFloat32.class));

            // Predict the class of each image in the batch and compute the loss
            Softmax<TFloat32> softmax =
                    tf.nn.softmax(
                            tf.math.add(
                                    tf.linalg.matMul(images, weights),
                                    biases
                            )
                    );
            Mean<TFloat32> crossEntropy =
                    tf.math.mean(
                            tf.math.neg(
                                    tf.reduceSum(
                                            tf.math.mul(labels, tf.math.log(softmax)),
                                            tf.array(1)
                                    )
                            ),
                            tf.array(0)
                    );

            // Back-propagate gradients to variables for training
            Optimizer optimizer = new GradientDescent(graph, LEARNING_RATE);
            Op minimize = optimizer.minimize(crossEntropy);

            // Compute the accuracy of the model
            Operand<TInt64> predicted = tf.math.argMax(softmax, tf.constant(1));
            Operand<TInt64> expected = tf.math.argMax(labels, tf.constant(1));
            Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.class), tf.array(0));

            // Run the graph
            try (Session session = new Session(graph)) {

                // Train the model
                for (int i = 0; i < 100; i++) {
                    try (TFloat32 batchImages = preprocessImages(imageNdArray);
                         TFloat32 batchLabels = preprocessLabels(labelNdArray)) {
                        session.runner()
                                .addTarget(minimize)
                                .feed(images.asOutput(), batchImages)
                                .feed(labels.asOutput(), batchLabels)
                                .run();
                    }
                }

                // Test the model
                try (TFloat32 testImages = preprocessImages(imageNdArray);
                     TFloat32 testLabels = preprocessLabels(labelNdArray);
                     TFloat32 accuracyValue = (TFloat32) session.runner()
                             .fetch(accuracy)
                             .feed(images.asOutput(), testImages)
                             .feed(labels.asOutput(), testLabels)
                             .run()
                             .get(0)) {
                    System.out.println("Accuracy: " + accuracyValue.getFloat());

                    saveModel(graph, "/Users/gregor/Desktop");
                }
            }
        }
    }


    private static TFloat32 preprocessImages(ByteNdArray rawImages) {
        Ops tf = Ops.create();

        // Flatten images in a single dimension and normalize their pixels as floats.
        long imageSize = rawImages.get(0).shape().size();
        return tf.math.div(
                tf.reshape(
                        tf.dtypes.cast(tf.constant(rawImages), TFloat32.class), tf.array(-1L, imageSize)),
                tf.constant(255.0f)
        ).asTensor();
    }

    private static TFloat32 preprocessLabels(ByteNdArray rawLabels) {
        Ops tf = Ops.create();
        // Map labels to one hot vectors where only the expected predictions as a value of 1.0
        return tf.oneHot(
                tf.constant(rawLabels),
                tf.constant(classDirs.length),
                tf.constant(1.0f),
                tf.constant(0.0f)
        ).asTensor();
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

    private static Tensor preprocessImage(BufferedImage img) {
        int targetWidth = 224;
        int targetHeight = 224;

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

    private static Tensor preprocessLabel(int classLabel, int numClasses) {
        float[] labelArray = new float[numClasses];
        labelArray[classLabel] = 1.0f;
        return TFloat32.tensorOf(StdArrays.ndCopyOf(labelArray));
    }

    private static void saveModel(Graph graph, String exportDir) throws IOException {
        // Fetch the graph definition
        GraphDef graphDef = GraphDef.parseFrom(graph.toGraphDef().toByteArray());

        // Save the frozen graph
        Files.write(Paths.get(exportDir, "/model.pb"), graphDef.toByteArray());

        System.out.println("Model saved to " + exportDir + "/model.pb");
    }
}