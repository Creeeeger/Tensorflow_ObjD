package org.object_d;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TUint8;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class image {
    public static void main(String[] args) throws IOException {
        // Load images
        String dataDir = "/Users/gregor/IdeaProjects/U6_CS_TENSORFLOW/flower_photos/";
        int imgHeight = 180;
        int imgWidth = 180;

        List<Tensor> imageTensors = new ArrayList<>();
        List<Long> labels = new ArrayList<>();

        String[] classNames = {"daisy", "dandelion", "roses", "sunflowers", "tulips"};
        for (int label = 0; label < classNames.length; label++) {
            String className = classNames[label];
            File dir = new File(Paths.get(dataDir, className).toString());
            for (File file : Objects.requireNonNull(dir.listFiles())) {
                BufferedImage img = ImageIO.read(new File("/Users/gregor/IdeaProjects/U6_CS_TENSORFLOW/flower_photos/roses/4910094611_8c7170fc95_n.jpg"));

                // Convert image to RGB if necessary
                BufferedImage rgbImg = new BufferedImage(imgWidth, imgHeight, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = rgbImg.createGraphics();
                g.drawImage(img, 0, 0, imgWidth, imgHeight, null);
                g.dispose();
                img = rgbImg;

                float[][][] imageArray = new float[imgHeight][imgWidth][3];
                for (int y = 0; y < imgHeight; y++) {
                    for (int x = 0; x < imgWidth; x++) {
                        int rgb = img.getRGB(x, y);
                        imageArray[y][x][0] = (rgb >> 16) & 0xFF;
                        imageArray[y][x][1] = (rgb >> 8) & 0xFF;
                        imageArray[y][x][2] = rgb & 0xFF;
                    }
                }

                imageTensors.add(Tensor.of(TUint8.class, Shape.of(imgHeight, imgWidth, 3)));
                labels.add((long) label);
            }
        }

        // Your further code to handle imageTensors and labels
    }
}