package org.object_d;

import nu.pattern.OpenCV;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class trained_detector extends JFrame {

    //Initialise the elements
    public static File tensor_file = new File("/");
    static JLabel Tensor_name, image_name, output_name, img;
    static JButton image_select, tensor_select, predict;
    static File image_file;

    public trained_detector() {
        //Create the layout
        setLayout(new BorderLayout(10, 10)); // Use BorderLayout with spacing
        JPanel detectorPanel = new JPanel();
        detectorPanel.setLayout(new BoxLayout(detectorPanel, BoxLayout.Y_AXIS));
        detectorPanel.setBorder(BorderFactory.createTitledBorder("Detector from previously created models")); // Add border with title

        //Set up the components
        Tensor_name = new JLabel("Tensor file");
        tensor_select = new JButton("Select Tensor file");

        image_name = new JLabel("Image file");
        image_select = new JButton("Select image file");
        image_select.setEnabled(false);

        output_name = new JLabel("Predicted class");

        predict = new JButton("Predict");
        predict.setEnabled(false);

        // Create a dummy image placeholder
        BufferedImage placeholderImage = new BufferedImage(200, 200, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = placeholderImage.createGraphics();
        g2d.setColor(Color.GRAY);
        g2d.fillRect(0, 0, 200, 200);
        g2d.setColor(Color.BLACK);
        g2d.drawString("Image comes here", 50, 100);
        g2d.dispose();
        ImageIcon dummyImage = new ImageIcon(placeholderImage);
        img = new JLabel(dummyImage);

        // Add space between components
        detectorPanel.add(Tensor_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(tensor_select);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(image_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(image_select);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(img);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(predict);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(output_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        //Add the button actions
        image_select.addActionListener(new event_select_image(img));
        tensor_select.addActionListener(new event_select_tensor());
        predict.addActionListener(new event_predict());

        add(detectorPanel, BorderLayout.CENTER);
    }

    public static TFloat32 image_preparation(File ImageFile) throws IOException {
        OpenCV.loadLocally();
        int targetWidth = 1024;
        int targetHeight = 1024;

        BufferedImage img = ImageIO.read(ImageFile);

        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(img, 0, 0, targetWidth, targetHeight, null);
        FloatNdArray imageData = NdArrays.ofFloats(Shape.of(1, targetHeight, targetWidth, 3));

        float[][][] imageArray = new float[targetHeight][targetWidth][3];  // Assuming 3 channels
        for (int x = 0; x < targetHeight; x++) {
            for (int y = 0; y < targetWidth; y++) {
                int rgb = resizedImage.getRGB(x, y);
                imageArray[x][y][0] = ((rgb >> 16) & 0xFF) / 255.0f;  // Red
                imageArray[x][y][1] = ((rgb >> 8) & 0xFF) / 255.0f;   // Green
                imageArray[x][y][2] = (rgb & 0xFF) / 255.0f;          // Blue
            }
        }

        // Fill the image tensor
        for (int i = 0; i < targetHeight; i++) {
            for (int j = 0; j < targetWidth; j++) {
                for (int k = 0; k < 3; k++) {
                    imageData.setFloat(imageArray[i][j][k], 0, i, j, k);
                }
            }
        }

        return (TFloat32.tensorOf(imageData));
    }

    public static void detect() throws IOException {
        try (SavedModelBundle model = SavedModelBundle.load(tensor_file.getPath(), "serve")) {
            TFloat32 imageTensor = image_preparation(image_file);

            // Fetch the outputs
            TFloat32 classOutput = (TFloat32) model.session().forceInitialize()
                    .runner()
                    .feed("input", imageTensor)
                    .fetch("class_output")
                    .run()
                    .get(0);

            System.out.println(classOutput.getFloat() + " class output probability");
        }
    }

    public static class event_select_image implements ActionListener {
        JLabel imageLabel;

        public event_select_image(JLabel imageLabel) {
            this.imageLabel = imageLabel;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();
                try {
                    ImageIcon icon = new ImageIcon(selectedFile.getPath());
                    Image originalImage = icon.getImage();
                    int desiredHeight = 300;
                    int desiredWidth = 400;
                    Image scaledImage = originalImage.getScaledInstance(desiredWidth, desiredHeight, Image.SCALE_SMOOTH);
                    ImageIcon scaledIcon = new ImageIcon(scaledImage);
                    imageLabel.setIcon(scaledIcon);
                    image_name.setText(selectedFile.getPath());
                    image_file = selectedFile;
                    predict.setEnabled(true);

                } catch (Exception ex) {
                    throw new RuntimeException(ex);
                }
            }
        }
    }

    public static class event_select_tensor implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                tensor_file = fileChooser.getSelectedFile();
                Tensor_name.setText(tensor_file.getPath());
            }
            System.out.println("Model loaded");
            image_select.setEnabled(true);
        }
    }

    public static class event_predict implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                detect();
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}