package org.object_d;

import nu.pattern.OpenCV;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.exceptions.TFInvalidArgumentException;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class trained_detector extends JFrame {
    // Create a File object to represent the path for a tensor file.
    static File tensor_file = new File(System.getProperty("user.dir") + File.separator + "tensor_file");  // Initially set to the current working directory
    static JLabel Tensor_name;       // Tensor_name label to display the name of the tensor file
    static JLabel image_name;        // image_name label to show the name of the selected image file
    static JLabel output_name;       // output_name label to display the name of the output
    static JButton image_select;     // JButton for selecting an image file
    static JButton predict;          // JButton to initiate the prediction process
    static File image_file;          // File object to store the selected image file

    public trained_detector() {
        // Create the layout using BorderLayout with 10px spacing
        setLayout(new BorderLayout(10, 10));

        // Create a panel for the detector and set its layout to BoxLayout with a vertical orientation
        JPanel detectorPanel = new JPanel();
        detectorPanel.setLayout(new BoxLayout(detectorPanel, BoxLayout.Y_AXIS));

        // Add a titled border to the panel
        detectorPanel.setBorder(BorderFactory.createTitledBorder("Detector from previously created models"));

        // Initialize labels and buttons for selecting tensor and image files
        Tensor_name = new JLabel("Tensor file");
        JButton tensor_select = new JButton("Select Tensor file");

        image_name = new JLabel("Image file");
        image_select = new JButton("Select image file");
        image_select.setEnabled(false); // Initially disabled since tensor file is selected first

        // Label for displaying the predicted class
        output_name = new JLabel("Predicted class: ");

        // Initialize the predict button, but keep it disabled initially
        predict = new JButton("Predict");
        predict.setEnabled(false); // Enabled only after files are selected

        // Create a dummy image placeholder for the user interface
        BufferedImage placeholderImage = new BufferedImage(200, 200, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = placeholderImage.createGraphics();
        g2d.setColor(Color.GRAY);
        g2d.fillRect(0, 0, 200, 200); // Fill a gray rectangle
        g2d.setColor(Color.BLACK);
        g2d.drawString("Image comes here", 50, 100); // Display placeholder text
        g2d.dispose(); // Dispose of the graphics object
        ImageIcon dummyImage = new ImageIcon(placeholderImage); // Convert image to an icon
        JLabel img = new JLabel(dummyImage); // Set the placeholder image in a JLabel

        // Add components to the detector panel with spacing between them
        detectorPanel.add(Tensor_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Adds 5px vertical space
        detectorPanel.add(tensor_select);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(image_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(image_select);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(img); // Placeholder image added
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(predict);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(output_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        // Add event listeners for button actions
        image_select.addActionListener(new event_select_image(img)); // Select image when clicked
        tensor_select.addActionListener(new event_select_tensor()); // Select tensor when clicked
        predict.addActionListener(new event_predict()); // Perform prediction when clicked

        // Add the panel to the center of the layout
        add(detectorPanel, BorderLayout.CENTER);
    }

    /**
     * Prepares the input image by resizing it to a fixed size and normalizing its RGB values.
     *
     * @param ImageFile The image file to be prepared.
     * @return A tensor (TFloat32) representing the image data in a normalized format suitable for model prediction.
     * @throws IOException If an error occurs while reading the image file.
     */
    public static TFloat32 image_preparation(File ImageFile, int targetSize) throws IOException {
        // Load OpenCV library locally to handle image manipulation
        OpenCV.loadLocally();

        // Read the image file from disk
        BufferedImage img = ImageIO.read(ImageFile);

        // Create a new BufferedImage for resizing the original image to the target dimensions
        BufferedImage resizedImage = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(img, 0, 0, targetSize, targetSize, null); // Draw the image scaled to the new size

        // Create a tensor (NdArray) to hold the image data in the format [1, height, width, 3] (for RGB)
        FloatNdArray imageData = NdArrays.ofFloats(Shape.of(1, targetSize, targetSize, 3));

        // Convert the image pixel values to a 3D array representing RGB values normalized between 0 and 1
        float[][][] imageArray = new float[targetSize][targetSize][3];
        for (int x = 0; x < targetSize; x++) {
            for (int y = 0; y < targetSize; y++) {
                int rgb = resizedImage.getRGB(x, y); // Get the RGB value at each pixel
                imageArray[x][y][0] = ((rgb >> 16) & 0xFF) / 255.0f;  // Extract and normalize red component
                imageArray[x][y][1] = ((rgb >> 8) & 0xFF) / 255.0f;   // Extract and normalize green component
                imageArray[x][y][2] = (rgb & 0xFF) / 255.0f;          // Extract and normalize blue component
            }
        }

        // Fill the image tensor with the normalized RGB values from the array
        for (int i = 0; i < targetSize; i++) {
            for (int j = 0; j < targetSize; j++) {
                for (int k = 0; k < 3; k++) {
                    imageData.setFloat(imageArray[i][j][k], 0, i, j, k); // Set the float values in the tensor
                }
            }
        }

        // Return the prepared image tensor for prediction use
        return (TFloat32.tensorOf(imageData));
    }

    /**
     * Initiates the image detection process. Attempts detection with initial size
     * retries with new size if previous size failed
     *
     * @throws IOException if an error occurs while loading the model or processing the image.
     */
    public static void detect() throws IOException {
        try {
            // Attempt the detection with an initial size
            runDetection(224);
        } catch (TFInvalidArgumentException e) {
            // Parse the required height from the error message
            String errorMsg = e.getMessage();
            int newHeight = parseRequiredDimension(errorMsg);

            // Retry the process with the new dimensions
            System.out.printf("Retrying with dimensions: %s%n\n", newHeight);
            runDetection(newHeight);
        }
    }

    /**
     * Runs the image detection process using a given input size
     *
     * @param inputSize The size to which the image should be resized before feeding it to the model.
     * @throws IOException if an error occurs while loading the model or processing the image.
     */
    private static void runDetection(int inputSize) throws IOException {
        // Load the trained model from the directory specified by tensor_file
        try (SavedModelBundle model = SavedModelBundle.load(tensor_file.getPath(), "serve")) {
            try (Session session = model.session()) {
                // Prepare the image file by converting it to a tensor with the given input size
                TFloat32 imageTensor = image_preparation(image_file, inputSize);

                // Run the model session and fetch the output for class prediction
                TFloat32 classOutput = (TFloat32) session.runner()
                        .feed("input", imageTensor)
                        .fetch("class_output")
                        .run()
                        .get(0);

                // Process the class predictions
                processClassOutput(classOutput);
            }
        }
    }

    /**
     * Processes the class output from the model by identifying the class with the highest probability.
     *
     * @param classOutput The output tensor containing the predicted class probabilities.
     */
    private static void processClassOutput(TFloat32 classOutput) {
        int predictedClass = 0;
        float maxProbability = -1.0f;

        // Iterate through the class probabilities to find the best prediction
        for (int i = 0; i < classOutput.shape().get(1); i++) {
            float probability = classOutput.getFloat(0, i);
            if (probability > maxProbability) {
                maxProbability = probability;
                predictedClass = i;
            }

            // Print out the probability for each class
            System.out.printf("Class %s, probability: %.4f\n", i, probability);
        }

        // Output the final prediction
        System.out.printf("Final predicted class: %d with probability: %.4f%n", predictedClass, maxProbability);
        output_name.setText(String.format("Predicted class: %d with probability: %.4f", predictedClass, maxProbability));
    }

    /**
     * Parses the error message to extract the required dimension for the input size.
     *
     * @param errorMessage The error message containing the required dimension.
     * @return The required dimension as an integer, standard is 224 since multiple of 32 which is commonly used
     */
    private static int parseRequiredDimension(String errorMessage) {
        // Extract the required height from the error message
        String regex = "requires a multiple of (\\d+)";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(errorMessage);

        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1)); // Return the parsed height
        }

        // Default value if parsing fails
        return 224;
    }

    public static class event_select_image implements ActionListener {
        JLabel imageLabel;

        // Constructor accepts a JLabel where the image will be displayed
        public event_select_image(JLabel imageLabel) {
            this.imageLabel = imageLabel;
        }

        // This method is triggered when the user selects an image file
        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser(); // Open a file chooser dialog
            int returnValue = fileChooser.showOpenDialog(null);

            // Check if the user selected a file
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile(); // Get the selected file

                try {
                    // Create an ImageIcon from the selected file and resize it for display
                    ImageIcon icon = new ImageIcon(selectedFile.getPath());
                    Image originalImage = icon.getImage();
                    int desiredHeight = 300;
                    int desiredWidth = 400;

                    // Scale the image to the desired size smoothly
                    Image scaledImage = originalImage.getScaledInstance(desiredWidth, desiredHeight, Image.SCALE_SMOOTH);
                    ImageIcon scaledIcon = new ImageIcon(scaledImage);

                    // Set the scaled image on the provided JLabel
                    imageLabel.setIcon(scaledIcon);

                    // Update the image_name label and store the file path for further processing
                    image_name.setText(selectedFile.getPath());
                    image_file = selectedFile; // Assign the selected file to image_file

                    // Enable the 'Predict' button since an image has been selected
                    predict.setEnabled(true);

                } catch (Exception ex) {
                    // If something goes wrong (e.g., file is not an image), throw a runtime exception
                    throw new RuntimeException(ex);
                }
            }
        }
    }

    public static class event_select_tensor implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            // Create a file chooser dialog for selecting a directory (tensor file location)
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Limit selection to directories only

            // Show the file chooser dialog and store the user's selection
            int returnValue = fileChooser.showOpenDialog(null);

            // If the user approves the selection (i.e., clicks "Open")
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                // Get the selected directory as the tensor file
                tensor_file = fileChooser.getSelectedFile();

                // Update the label to display the selected tensor file's path
                Tensor_name.setText(tensor_file.getPath());
            }

            // Print a message to confirm the model (directory) has been loaded
            System.out.println("Model loaded");

            // Enable the image selection button now that the tensor file has been selected
            image_select.setEnabled(true);
        }
    }

    public static class event_predict implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // perform the prediction when the "Predict" button is pressed
            try {
                // Call the detect method to run the detection process
                detect();
            } catch (Exception ex) {
                // If an exception occurs during detection, throw a RuntimeException
                throw new RuntimeException(ex);
            }
        }
    }
}