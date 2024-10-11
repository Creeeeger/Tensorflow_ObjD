package org.object_d;

import org.tensorAction.detector;
import org.tensorflow.SavedModelBundle;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Objects;

import static org.object_d.config_handler.load_config;

public class Main_UI extends JFrame {
    // Static JLabel components for displaying various information and results
    public static JLabel label; // Label for general display
    public static JLabel img; // Label for displaying images
    public static JLabel image_path; // Label to show the image file path
    public static JLabel model_path; // Label to display the model file path
    public static JLabel result; // Label for showing results
    public static JLabel output_img; // Label for output images

    // Static JMenuBar and JMenu components for creating a menu interface
    public static JMenuBar menuBar; // Main menu bar
    public static JMenu file; // Menu for file-related actions
    public static JMenu model; // Menu for model-related actions
    public static JMenu database; // Menu for database operations
    public static JMenu model_trainer; // Menu for model training options
    public static JMenu detector_menu; // Menu for object detection options

    // Static JMenuItem components for individual menu actions
    public static JMenuItem exit; // Menu item for exiting the application
    public static JMenuItem load; // Menu item for loading a file
    public static JMenuItem load_database; // Menu item for loading a database
    public static JMenuItem reset_database; // Menu item for resetting the database
    public static JMenuItem db_utility; // Menu item for database utility options
    public static JMenuItem load_model; // Menu item for loading a model
    public static JMenuItem set_params; // Menu item for setting parameters
    public static JMenuItem restore_last; // Menu item for restoring the last state
    public static JMenuItem train_model; // Menu item for training a model
    public static JMenuItem self_detector; // Menu item for self-detection options
    public static JMenuItem save_manually; // Menu item for manual saving

    // Static JScrollPane for enabling scrolling of data
    public static JScrollPane data_scrollPane; // Scroll pane for displaying data

    // Static JPanel components for organizing the layout of the user interface
    public static JPanel leftPanel, rightPanel; // Panels for left and right boxes

    // Static File components for handling file paths
    public static File tensor_file = new File("/"); // Default tensor file path
    public static File prev_picture = new File("/"); // Default previous picture path

    // Static JButton component for triggering object detection
    public static JButton detect_objects; // Button for detecting objects

    // Static variables for training parameters
    public static int resolution; // Variable for image resolution
    public static int epochs; // Variable for number of training epochs
    public static int batch_size; // Variable for batch size during training
    public static float learning_rate; // Variable for learning rate

    // Static variable for handling the saved model bundle
    static SavedModelBundle savedModelBundle; // Bundle for managing the saved model

    public Main_UI() {
        // Set layout for the main UI as a horizontal grid with spacing
        setLayout(new GridLayout(1, 2, 10, 10));

        // Create left and right panels for the UI
        leftPanel = new JPanel();
        leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS)); // Vertical layout for left panel
        leftPanel.setBorder(BorderFactory.createTitledBorder("Detection Panel")); // Add title border

        rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS)); // Vertical layout for right panel
        rightPanel.setBorder(BorderFactory.createTitledBorder("Data Panel")); // Add title border

        // Add both panels to the main frame
        add(leftPanel);
        add(rightPanel);

        // Create a dummy image placeholder for display
        BufferedImage placeholderImage = new BufferedImage(200, 200, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = placeholderImage.createGraphics(); // Create graphics context
        g2d.setColor(Color.GRAY); // Set color to gray for the placeholder
        g2d.fillRect(0, 0, 200, 200); // Fill rectangle with gray color
        g2d.setColor(Color.BLACK); // Set color to black for text
        g2d.drawString("Image comes here", 50, 100); // Add placeholder text
        g2d.dispose(); // Dispose graphics context
        ImageIcon dummyImage = new ImageIcon(placeholderImage); // Create ImageIcon from placeholder

        // Left Panel Components
        label = new JLabel("Object detector"); // Title label for detection
        img = new JLabel(dummyImage); // Label to show the dummy image
        image_path = new JLabel("here comes the image path (select the actual image)"); // Label for image path
        model_path = new JLabel("here comes the model path (select the folder with the tensor file in it)"); // Label for model path
        result = new JLabel("Predicted results here"); // Label for predicted results
        output_img = new JLabel(dummyImage); // Label for output image

        // Add components to the left panel with spacing
        leftPanel.add(label);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Space between components
        leftPanel.add(img);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Space between components
        leftPanel.add(image_path);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Space between components
        leftPanel.add(model_path);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Space between components
        leftPanel.add(output_img);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Space between components
        leftPanel.add(result);

        // Button to trigger object detection
        detect_objects = new JButton("Recognise Objects");
        detect_objects.setEnabled(false); // Initially disabled until an image is loaded
        detect_objects.addActionListener(new detect_ev()); // Add action listener for button
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Space between components
        leftPanel.add(detect_objects); // Add button to left panel

        // Right Panel Components
        data_scrollPane = new JScrollPane(); // Create scroll pane for right panel
        rightPanel.add(data_scrollPane); // Add scroll pane to right panel

        // Menu Bar Configuration
        menuBar = new JMenuBar(); // Create menu bar
        setJMenuBar(menuBar); // Set the menu bar for the main frame

        // File Menu and its items
        file = new JMenu("File"); // Create "File" menu
        load = new JMenuItem("Load image"); // Menu item for loading an image
        load.addActionListener(new event_load(img)); // Add action listener
        load_model = new JMenuItem("Load a tensor model"); // Menu item for loading a model
        load_model.addActionListener(new event_load_tensor()); // Add action listener
        restore_last = new JMenuItem("Restore last config"); // Menu item for restoring last config
        restore_last.addActionListener(new event_restore_last()); // Add action listener
        save_manually = new JMenuItem("Save config"); // Menu item for saving configuration
        save_manually.addActionListener(new save_manu()); // Add action listener
        exit = new JMenuItem("Save and Exit"); // Menu item for exiting the application
        exit.addActionListener(new event_exit()); // Add action listener

        // Add file menu items to the file menu
        file.add(load);
        file.add(load_model);
        file.add(restore_last);
        file.add(save_manually);
        file.add(exit);

        // Add file menu to the menu bar
        menuBar.add(file);

        // Model Menu and its items
        model = new JMenu("Model"); // Create "Model" menu
        set_params = new JMenuItem("Set model parameters"); // Menu item for setting parameters
        set_params.addActionListener(new event_set_params()); // Add action listener
        model.add(set_params); // Add parameter setting item to the model menu
        menuBar.add(model); // Add model menu to the menu bar

        // Database Menu and its items
        database = new JMenu("Database"); // Create "Database" menu
        load_database = new JMenuItem("Load database"); // Menu item for loading a database
        load_database.addActionListener(new event_load_database()); // Add action listener
        reset_database = new JMenuItem("Reset database"); // Menu item for resetting database
        reset_database.addActionListener(new event_reset_database()); // Add action listener
        db_utility = new JMenuItem("Database utility"); // Menu item for database utilities
        db_utility.addActionListener(new event_database_utility()); // Add action listener
        database.add(load_database); // Add database loading item to the database menu
        database.add(reset_database); // Add reset database item to the database menu
        database.add(db_utility); // Add database utility item to the database menu
        menuBar.add(database); // Add database menu to the menu bar

        // Model Trainer Menu and its items
        model_trainer = new JMenu("Model creator"); // Create "Model creator" menu
        train_model = new JMenuItem("Train own models"); // Menu item for training own models
        train_model.addActionListener(new event_train()); // Add action listener
        model_trainer.add(train_model); // Add training item to the model trainer menu
        menuBar.add(model_trainer); // Add model trainer menu to the menu bar

        // Object Detection Menu and its items
        detector_menu = new JMenu("Object detection v2"); // Create "Object detection" menu
        self_detector = new JMenuItem("detect objects with own models"); // Menu item for detecting objects
        self_detector.addActionListener(new create_detector_window()); // Add action listener
        detector_menu.add(self_detector); // Add detection item to the detector menu
        menuBar.add(detector_menu); // Add detector menu to the menu bar
    }

    public static void main(String[] args) {
        // Create config file if it doesn't exist
        File config = new File("config.xml");
        if (!config.exists()) {
            org.object_d.config_handler.create_config(); // Call method to create config
            System.out.println("Config Created"); // Output confirmation
        }

        // Create database file if it doesn't exist
        File database = new File("results.db");
        if (!database.exists()) {
            database_handler.reset_init_db(); // Call method to initialize database
            System.out.println("Database created"); // Output confirmation
        }

        // Load configuration values
        String[][] values_load = load_config(); // Load config values
        setValues(values_load); // Set loaded values

        // Initialize and set up the main UI
        Main_UI gui = new Main_UI(); // Create instance of the Main_UI
        gui.setVisible(true); // Make the GUI visible
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Set default close operation
        gui.setSize(1200, 1000); // Set window size
        gui.setTitle("Object Detector UI"); // Set window title
    }

    public static void setValues(String[][] values_load) {
        for (String[] value : Objects.requireNonNull(values_load)) { // Iterate over each value
            System.out.println(value[0] + " " + value[1]); // Print key-value pairs

            switch (value[0]) {
                case "img_path": // Check for image path
                    File selectedFile = new File(value[1]); // Create file object
                    try {
                        JLabel imageLabel = img; // Get the image label
                        ImageIcon icon = new ImageIcon(selectedFile.getPath()); // Load image
                        Image originalImage = icon.getImage(); // Get original image
                        int desiredHeight = 300; // Desired height for scaling
                        int desiredWidth = 400; // Desired width for scaling
                        Image scaledImage = originalImage.getScaledInstance(desiredWidth, desiredHeight, Image.SCALE_SMOOTH); // Scale image
                        ImageIcon scaledIcon = new ImageIcon(scaledImage); // Create new icon
                        imageLabel.setIcon(scaledIcon); // Set scaled icon to label
                        prev_picture = selectedFile; // Store the previous picture
                        image_path.setText(prev_picture.getPath()); // Update image path label
                    } catch (Exception ex) { // Handle exceptions
                        if (ex.getClass() == NullPointerException.class) {
                            continue; // Ignore NullPointerException
                        } else {
                            throw new RuntimeException(ex); // Rethrow other exceptions
                        }
                    }
                    break;

                case "ts_path": // Check for tensor model path
                    try {
                        tensor_file = new File(value[1]); // Create tensor file object
                        model_path.setText(tensor_file.getPath()); // Update model path label
                        detect_objects.setEnabled(true); // Enable detection button
                        savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve"); // Load the model
                        System.out.println("Model loaded"); // Output confirmation
                    } catch (Exception ex) { // Handle exceptions
                        System.out.println("could not load tensor"); // Output error message
                    }
                    break;

                case "resolution": // Set resolution value
                    resolution = Integer.parseInt(value[1]); // Parse and set resolution
                    break;

                case "batch": // Set batch size
                    epochs = Integer.parseInt(value[1]); // Parse and set epochs
                    break;

                case "epochs": // Set epochs value
                    batch_size = Integer.parseInt(value[1]); // Parse and set batch size
                    break;

                case "learning": // Set learning rate
                    learning_rate = Float.parseFloat(value[1]); // Parse and set learning rate
                    break;

                default: // Handle unknown settings
                    System.out.println("Unknown setting: " + value[0]); // Output unknown setting message
                    break;
            }
        }
    }

    public void save_reload_config(int res, int epo, int bat, float lea, String pic, String ten) {
        System.out.println(res); // Output resolution for debugging
        String[][] values = { // Create values array for saving config
                {"img_path", pic}, // Image path
                {"ts_path", ten}, // Tensor model path
                {"resolution", String.valueOf(res)}, // Resolution
                {"batch", String.valueOf(epo)}, // Epochs
                {"epochs", String.valueOf(bat)}, // Batch size
                {"learning", String.valueOf(lea)} // Learning rate
        };
        config_handler.save_config(values); // Save configuration

        String[][] values_load = load_config(); // Load configuration values
        setValues(values_load); // Set loaded values
    }

    public static class save_manu implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event
            String[][] values = { // Create values array for saving config
                    {"img_path", prev_picture.getPath()}, // Image path
                    {"ts_path", tensor_file.getPath()}, // Tensor model path
                    {"resolution", String.valueOf(resolution)}, // Resolution
                    {"batch", String.valueOf(epochs)}, // Epochs
                    {"epochs", String.valueOf(batch_size)}, // Batch size
                    {"learning", String.valueOf(learning_rate)} // Learning rate
            };

            config_handler.save_config(values); // Save configuration
        }
    }

    public static class create_detector_window implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event
            trained_detector gui = new trained_detector(); // Create new detector window
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set close operation
            gui.setVisible(true); // Make window visible
            gui.setTitle("Object detector for own models"); // Set window title
            gui.setSize(600, 600); // Set window size
            gui.setLocation(100, 100); // Set window location
        }
    }

    public static class event_train implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event
            Trainer gui = new Trainer(); // Create new trainer window
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set close operation
            gui.setVisible(true); // Make window visible
            gui.setSize(1400, 900); // Set window size
            gui.setLocation(100, 100); // Set window location
            gui.setTitle("Model trainer"); // Set window title
        }
    }

    public static class event_set_params implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event
            model_param gui = new model_param( // Create new model parameter settings window
                    prev_picture.getPath(), // Pass previous picture path
                    tensor_file.getPath(), // Pass tensor model path
                    resolution, // Pass current resolution
                    epochs, // Pass current epochs
                    batch_size, // Pass current batch size
                    learning_rate // Pass current learning rate
            );

            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set close operation
            gui.setVisible(true); // Make window visible
            gui.setSize(1100, 550); // Set window size
            gui.setLocation(100, 100); // Set window location
            gui.setTitle("Model Parameter Settings"); // Set window title
        }
    }

    public static class event_exit implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event
            String[][] values = { // Create an array to hold configuration values
                    {"img_path", prev_picture.getPath()}, // Store previous picture path
                    {"ts_path", tensor_file.getPath()}, // Store tensor model path
                    {"resolution", String.valueOf(resolution)}, // Store resolution
                    {"batch", String.valueOf(epochs)}, // Store batch size
                    {"epochs", String.valueOf(batch_size)}, // Store epochs
                    {"learning", String.valueOf(learning_rate)} // Store learning rate
            };

            config_handler.save_config(values); // Save the configuration values

            System.exit(0); // Exit the application
        }
    }

    public static class detect_ev implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event
            // Load up the model bundle from the tensor file
            savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
            System.out.println("Model loaded");

            // Get the classification results for the image
            ArrayList<detector.entry> result_array;
            result_array = org.tensorAction.detector.classify(image_path.getText(), savedModelBundle);

            // Load and display the resulting image
            File imagePath = new File(result_array.getLast().getImagePath());
            ImageIcon icon = new ImageIcon(String.valueOf(imagePath));

            Image originalImage = icon.getImage(); // Get the original image
            Image scaledImage = originalImage.getScaledInstance(400, 300, Image.SCALE_SMOOTH); // Scale image
            ImageIcon scaledIcon = new ImageIcon(scaledImage);
            output_img.setIcon(scaledIcon); // Set the scaled image icon to the output label

            StringBuilder dataString = new StringBuilder(); // Initialize an empty data string builder

            try {
                for (int i = 0; i < result_array.size() - 1; i++) { // Loop until the second last element since the last is the image path
                    detector.entry entry = result_array.get(i); // Get each classification entry
                    dataString.append(entry.getLabel()).append(" ").append(entry.getPercentage()).append("%, "); // Append label and percentage
                }

                result_array.removeLast(); // Remove the last entry since it's just storing the image path

                if (!result_array.isEmpty()) { // Check that there is data to save
                    database_handler.addData(result_array); // Save data to the database
                }
            } catch (Exception e1) {
                throw new RuntimeException(e1); // Handle exceptions
            }
            result.setText(dataString.toString()); // Display results in the result label
        }
    }

    public static class event_load_tensor implements ActionListener { // Handles loading of tensor model

        @Override
        public void actionPerformed(ActionEvent e) { // Action performed when the menu item is clicked
            JFileChooser fileChooser = new JFileChooser(); // Create a file chooser
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set selection mode to directories only
            int returnValue = fileChooser.showOpenDialog(null); // Show the open dialog

            if (returnValue == JFileChooser.APPROVE_OPTION) { // Check if the user approved the selection
                tensor_file = fileChooser.getSelectedFile(); // Get the selected file (directory)
                model_path.setText(tensor_file.getPath()); // Display the selected path in the model path label
                detect_objects.setEnabled(true); // Enable the "Recognise Objects" button
            }

            try {
                // Load the saved model bundle from the selected tensor file
                savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
                System.out.println("Model loaded"); // Log successful model loading
            } catch (Exception ex) {
                throw new RuntimeException(ex); // Handle exceptions by throwing a runtime exception
            }
        }
    }

    public static class event_reset_database implements ActionListener { // Handles resetting the database

        @Override
        public void actionPerformed(ActionEvent e) { // Action performed when the menu item is clicked
            // Create an instance of the reset confirmation dialog
            reset_confirmation gui = new reset_confirmation();

            // Set the dialog's properties
            gui.setVisible(true); // Make the dialog visible
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Close the dialog without exiting the application
            gui.setLocation(100, 100); // Set the location of the dialog
            gui.setSize(500, 300); // Set the size of the dialog

            // Update the main UI label to indicate that confirmation is pending
            label.setText("wait for confirmation");
        }
    }

    public static class event_load implements ActionListener { // Handles loading an image and displaying it
        JLabel imageLabel; // JLabel to display the loaded image

        public event_load(JLabel imageLabel) { // Constructor receives the JLabel where the image will be displayed
            this.imageLabel = imageLabel;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser(); // Initialize a file chooser
            int returnValue = fileChooser.showOpenDialog(null); // Show the file chooser dialog

            if (returnValue == JFileChooser.APPROVE_OPTION) { // If the user selects a file
                File selectedFile = fileChooser.getSelectedFile(); // Get the selected file
                try {
                    // Create an ImageIcon from the selected file
                    ImageIcon icon = new ImageIcon(selectedFile.getPath());
                    Image originalImage = icon.getImage(); // Get the image
                    int desiredHeight = 300; // Desired height for scaling
                    int desiredWidth = 400;  // Desired width for scaling
                    // Scale the image
                    Image scaledImage = originalImage.getScaledInstance(desiredWidth, desiredHeight, Image.SCALE_SMOOTH);
                    ImageIcon scaledIcon = new ImageIcon(scaledImage); // Create a new icon with the scaled image
                    imageLabel.setIcon(scaledIcon); // Set the scaled image to the JLabel
                    prev_picture = selectedFile; // Save the selected file for future reference
                    image_path.setText(prev_picture.getPath()); // Update the image path label
                } catch (Exception ex) {
                    throw new RuntimeException(ex); // Throw exception if there's an issue loading the image
                }
            }
        }
    }

    public static class event_database_utility implements ActionListener { // Handles opening the Database Utility window
        @Override
        public void actionPerformed(ActionEvent e) {
            // Initialize and configure the database utility window
            database_utility gui = new database_utility();
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Close the window without exiting the application
            gui.setSize(1400, 800); // Set the window size
            gui.setTitle("Database utility"); // Set the window title
            gui.setVisible(true); // Make the window visible
            gui.setLocation(100, 100); // Set the window location on the screen
        }
    }

    public static class event_restore_last implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            // Remove all components from the rightPanel to refresh with new data
            rightPanel.removeAll();

            // Load the configuration values
            String[][] values = load_config();
            setValues(values); // Set these values into the relevant UI components

            // Create a non-editable table model for displaying the loaded configuration
            DefaultTableModel nonEditableModel = new DefaultTableModel(values, new Object[]{"Name", "Value"}) {
                @Override
                public boolean isCellEditable(int row, int column) {
                    return false;  // Disable cell editing
                }
            };

            // Create a table with the non-editable model
            JTable table = new JTable(nonEditableModel);

            // Enable row selection, but disable column selection
            table.setEnabled(true);
            table.setRowSelectionAllowed(true);
            table.setColumnSelectionAllowed(false);

            // Create a scroll pane and add the table to it
            JScrollPane scrollPane = new JScrollPane(table);

            // Add the scroll pane to the right panel
            rightPanel.add(scrollPane);

            // Revalidate and repaint the panel to display the updated components
            rightPanel.revalidate();
            rightPanel.repaint();

            // Enable the "detect objects" button after restoring the last configuration
            detect_objects.setEnabled(true);
        }
    }

    public static class event_load_database implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            // First, remove any previously added components from the rightPanel
            rightPanel.removeAll();

            // Retrieve the data from the database
            String[][] data = database_handler.readDatabase();

            // Create a non-editable table model with column headers: "Name", "date", and "amount"
            DefaultTableModel nonEditableModel = new DefaultTableModel(data, new Object[]{"Name", "date", "amount"}) {
                @Override
                public boolean isCellEditable(int row, int column) {
                    return false;  // Disable editing for all cells
                }
            };

            // Create a JTable with the non-editable model
            JTable table = new JTable(nonEditableModel);

            // Enable row selection but disable column selection
            table.setEnabled(true);  // Allow row selection
            table.setRowSelectionAllowed(true);  // Allow rows to be selected
            table.setColumnSelectionAllowed(false);  // Disable column selection

            // Create a JScrollPane and add the JTable to it
            JScrollPane scrollPane = new JScrollPane(table);

            // Add the scroll pane (containing the table) to the rightPanel
            rightPanel.add(scrollPane);

            // Revalidate and repaint the panel to reflect the changes
            rightPanel.revalidate();
            rightPanel.repaint();
        }
    }
}