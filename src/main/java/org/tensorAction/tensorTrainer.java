package org.tensorAction;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import javax.imageio.IIOException;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class tensorTrainer {
    public static void main(String[] args) throws IOException {
        String dataDir = "/Users/gregor/Desktop/Tensorflow_ObjD/flower_photos";
        List<Tensor> imageTensors = new ArrayList<>();
        List<Tensor> labelTensors = new ArrayList<>();

        File[] classDirs = new File(dataDir).listFiles(File::isDirectory);
        Tensor imageTensor = null;
        Tensor labelTensor = null;
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
}