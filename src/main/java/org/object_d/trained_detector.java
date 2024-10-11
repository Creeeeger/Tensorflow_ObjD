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
    // Create a File object to represent the path for a tensor file.
    // Initially, the path is set to the root directory ("/"),
    public static File tensor_file = new File("/");

    // Static GUI components for the application interface
    // JLabel is used to display text and images in the GUI

    // Tensor_name label to display the name of the tensor file
    static JLabel Tensor_name;

    // image_name label to show the name of the selected image file
    static JLabel image_name;

    // output_name label to display the name of the output
    static JLabel output_name;

    // img label to display an image in the GUI
    static JLabel img;

    // JButton for selecting an image file. This button will trigger a file chooser dialog to select an image.
    static JButton image_select;

    // JButton for selecting a tensor file. This button will also trigger a file chooser dialog to select a tensor file.
    static JButton tensor_select;

    // JButton to initiate the prediction process. This button will call the method to make predictions based on the selected image and tensor.
    static JButton predict;

    // File object to store the selected image file. This will be updated when the user selects an image.
    static File image_file;

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
        tensor_select = new JButton("Select Tensor file");

        image_name = new JLabel("Image file");
        image_select = new JButton("Select image file");
        image_select.setEnabled(false); // Initially disabled since tensor file is selected first

        // Label for displaying the predicted class; contains a note about potential issues with variable initialization
        output_name = new JLabel("Predicted class -- Since we made the model in java and ts2x removed since 3 Years the init function this probably wont work and fail due to failed variable initialization");

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
        img = new JLabel(dummyImage); // Set the placeholder image in a JLabel

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

    public static TFloat32 image_preparation(File ImageFile) throws IOException {
        // Load OpenCV library locally to handle image manipulation
        OpenCV.loadLocally();

        // Set target dimensions for resizing the image
        int targetWidth = 1024;
        int targetHeight = 1024;

        // Read the image file from disk
        BufferedImage img = ImageIO.read(ImageFile);

        // Create a new BufferedImage for resizing the original image to the target dimensions
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(img, 0, 0, targetWidth, targetHeight, null); // Draw the image scaled to the new size

        // Create a tensor (NdArray) to hold the image data in the format [1, height, width, 3] (for RGB)
        FloatNdArray imageData = NdArrays.ofFloats(Shape.of(1, targetHeight, targetWidth, 3));

        // Convert the image pixel values to a 3D array representing RGB values normalized between 0 and 1
        float[][][] imageArray = new float[targetHeight][targetWidth][3];
        for (int x = 0; x < targetHeight; x++) {
            for (int y = 0; y < targetWidth; y++) {
                int rgb = resizedImage.getRGB(x, y); // Get the RGB value at each pixel
                imageArray[x][y][0] = ((rgb >> 16) & 0xFF) / 255.0f;  // Extract and normalize red component
                imageArray[x][y][1] = ((rgb >> 8) & 0xFF) / 255.0f;   // Extract and normalize green component
                imageArray[x][y][2] = (rgb & 0xFF) / 255.0f;          // Extract and normalize blue component
            }
        }

        // Fill the image tensor with the normalized RGB values from the array
        for (int i = 0; i < targetHeight; i++) {
            for (int j = 0; j < targetWidth; j++) {
                for (int k = 0; k < 3; k++) {
                    imageData.setFloat(imageArray[i][j][k], 0, i, j, k); // Set the float values in the tensor
                }
            }
        }

        // Return the prepared image tensor for prediction use
        return (TFloat32.tensorOf(imageData));
    }

    public static void detect() throws IOException {
        // Load the trained model from the directory specified by tensor_file, using the 'serve' tag.
        try (SavedModelBundle model = SavedModelBundle.load(tensor_file.getPath(), "serve")) {

            // Prepare the image file by converting it to a tensor that can be fed into the model.
            TFloat32 imageTensor = image_preparation(image_file);

            // Run the model session and fetch the output for class prediction.
            // 'input' refers to the model's input tensor name, and 'class_output' is the tensor
            // that will contain the class probabilities.
            TFloat32 classOutput = (TFloat32) model.session().forceInitialize()
                    .runner()
                    .feed("input", imageTensor)   // Feed the prepared image tensor to the model input
                    .fetch("class_output")        // Fetch the predicted class output
                    .run()
                    .get(0);                      // Get the first output, which is the class prediction

            // Print the predicted class probability.
            System.out.println(classOutput.getFloat() + " class output probability");
        }
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