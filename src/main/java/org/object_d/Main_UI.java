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
    // Static variables for training parameters
    public static int resolution; // Variable for image resolution
    public static int batch_size; // Variable for  batch size
    public static int epochs; // Variable for epochs
    public static float learning_rate; // Variable for learning rate
    // Static JLabel components for displaying various information and results
    static JLabel label; // Label for general display
    static JLabel img; // Label for displaying images
    static JLabel image_path; // Label to show the image file path
    static JLabel model_path; // Label to display the model file path
    static JLabel result; // Label for showing results
    static JLabel output_img; // Label for output images
    // Static JPanel component for organizing the layout of the user interface
    static JPanel rightPanel; // Panels for right boxes
    // Static File components for handling file paths
    static File tensor_file = new File(System.getProperty("user.dir") + File.separator + "tensor_file"); // Default tensor file path
    static File prev_picture = new File(System.getProperty("user.dir") + File.separator + "prev_picture"); // Default previous picture path
    // Static JButton component for triggering object detection
    static JButton detect_objects; // Button for detecting objects
    // Static variable for handling the saved model bundle
    static SavedModelBundle savedModelBundle; // Bundle for managing the saved model

    public Main_UI() {
        applyDarkMode(); // Apply dark mode settings

        // Set layout for the main UI as a horizontal grid with spacing
        setLayout(new GridLayout(1, 2, 10, 10));

        // Create left and right panels for the UI
        JPanel leftPanel = new JPanel();
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
        JScrollPane data_scrollPane = new JScrollPane(); // Create scroll pane for right panel
        rightPanel.add(data_scrollPane); // Add scroll pane to right panel

        // Menu Bar Configuration
        JMenuBar menuBar = new JMenuBar(); // Create menu bar
        setJMenuBar(menuBar); // Set the menu bar for the main frame

        // File Menu and its items
        JMenu file = new JMenu("File"); // Create "File" menu
        JMenuItem load = new JMenuItem("Load image"); // Menu item for loading an image
        load.addActionListener(new event_load(img)); // Add action listener
        JMenuItem load_model = new JMenuItem("Load a tensor model"); // Menu item for loading a model
        load_model.addActionListener(new event_load_tensor()); // Add action listener
        JMenuItem restore_last = new JMenuItem("Restore last config"); // Menu item for restoring last config
        restore_last.addActionListener(new event_restore_last()); // Add action listener
        JMenuItem save_manually = new JMenuItem("Save config"); // Menu item for saving configuration
        save_manually.addActionListener(new save_manu()); // Add action listener
        JMenuItem exit = new JMenuItem("Save and Exit"); // Menu item for exiting the application
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
        JMenu model = new JMenu("Model"); // Create "Model" menu
        JMenuItem set_params = new JMenuItem("Set model parameters"); // Menu item for setting parameters
        set_params.addActionListener(new event_set_params()); // Add action listener
        model.add(set_params); // Add parameter setting item to the model menu
        menuBar.add(model); // Add model menu to the menu bar

        // Database Menu and its items
        JMenu database = new JMenu("Database"); // Create "Database" menu
        JMenuItem load_database = new JMenuItem("Load database"); // Menu item for loading a database
        load_database.addActionListener(new event_load_database()); // Add action listener
        JMenuItem reset_database = new JMenuItem("Reset database"); // Menu item for resetting database
        reset_database.addActionListener(new event_reset_database()); // Add action listener
        JMenuItem db_utility = new JMenuItem("Database utility"); // Menu item for database utilities
        db_utility.addActionListener(new event_database_utility()); // Add action listener
        database.add(load_database); // Add database loading item to the database menu
        database.add(reset_database); // Add reset database item to the database menu
        database.add(db_utility); // Add database utility item to the database menu
        menuBar.add(database); // Add database menu to the menu bar

        // Model Trainer Menu and its items
        JMenu model_trainer = new JMenu("Model creator"); // Create "Model creator" menu
        JMenuItem train_model = new JMenuItem("Train own models"); // Menu item for training own models
        train_model.addActionListener(new event_train()); // Add action listener
        model_trainer.add(train_model); // Add training item to the model trainer menu
        menuBar.add(model_trainer); // Add model trainer menu to the menu bar

        // Object Detection Menu and its items
        JMenu detector_menu = new JMenu("Object detection v2"); // Create "Object detection" menu
        JMenuItem self_detector = new JMenuItem("Detect Objects with own trained model"); // Menu item for detecting objects
        self_detector.addActionListener(new create_detector_window()); // Add action listener
        detector_menu.add(self_detector); // Add detection item to the detector menu

        JMenuItem CNNPlayground = new JMenuItem("Create your own custom Cnn GUI");
        CNNPlayground.addActionListener(new event_launch_CNN_playground());
        detector_menu.add(CNNPlayground);
        menuBar.add(detector_menu); // Add detector menu to the menu bar
    }

    public static void main(String[] args) {
        // Create config file if it doesn't exist
        File config = new File(System.getProperty("user.dir") + File.separator + "config.xml");
        if (!config.exists()) {
            org.object_d.config_handler.create_config(); // Call method to create config
            System.out.println("Config Created"); // Output confirmation
        }

        // Create database file if it doesn't exist
        File database = new File(System.getProperty("user.dir") + File.separator + "results.db");
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

    /**
     * Sets the values for various configurations based on the input values array.
     *
     * @param values_load A 2D array containing key-value pairs representing configuration values.
     */
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
                    batch_size = Integer.parseInt(value[1]); // Parse and set batch size
                    break;

                case "epochs": // Set epochs value
                    epochs = Integer.parseInt(value[1]); // Parse and set epochs
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

    // Method for changing the colour of items into dark mode
    private void applyDarkMode() {
        UIManager.put("Table.background", Color.DARK_GRAY); // Background of the table (rows and cells)
        UIManager.put("Table.foreground", Color.WHITE); // Text color for table cells
        UIManager.put("Table.gridColor", Color.GRAY); // Color of the grid lines between cells
        UIManager.put("Table.selectionBackground", Color.GRAY); // Background color for selected rows
        UIManager.put("Table.selectionForeground", Color.WHITE); // Text color for selected rows
        UIManager.put("Table.headerBackground", Color.BLACK); // Background color for table header
        UIManager.put("Table.headerForeground", Color.WHITE); // Text color for table header
        UIManager.put("Table.headerFont", new Font("Arial", Font.BOLD, 12)); // Font for the table header
        UIManager.put("Table.cellFont", new Font("Arial", Font.PLAIN, 12)); // Font for table cells
        UIManager.put("Table.rowHeight", 30); // Row height for better readability

        UIManager.put("FileChooser.background", Color.DARK_GRAY); // Background of the entire file chooser
        UIManager.put("FileChooser.foreground", Color.WHITE); // Text color
        UIManager.put("FileChooser.selectionBackground", Color.GRAY); // Background of selected file
        UIManager.put("FileChooser.selectionForeground", Color.WHITE); // Text color of selected file
        UIManager.put("FileChooser.listViewBackground", Color.DARK_GRAY); // Background of the file list
        UIManager.put("FileChooser.listViewForeground", Color.WHITE); // Text color in the file list
        UIManager.put("FileChooser.listViewSelectionBackground", Color.GRAY); // Highlight color for selection
        UIManager.put("FileChooser.listViewSelectionForeground", Color.WHITE); // Highlighted text color
        UIManager.put("FileChooser.controlPanelBackground", Color.DARK_GRAY); // Background of control buttons
        UIManager.put("FileChooser.controlPanelForeground", Color.WHITE); // Text color for control buttons
        UIManager.put("FileChooser.buttonBackground", Color.GRAY); // Button background
        UIManager.put("FileChooser.buttonForeground", Color.WHITE); // Button text color
        UIManager.put("FileChooser.buttonHighlight", Color.LIGHT_GRAY); // Button highlight when pressed
        UIManager.put("FileChooser.border", BorderFactory.createLineBorder(Color.GRAY)); // Border for the file chooser
        UIManager.put("FileChooser.directoryBackground", Color.DARK_GRAY);
        UIManager.put("FileChooser.directoryForeground", Color.WHITE);
        UIManager.put("FileChooser.listBackground", Color.DARK_GRAY);
        UIManager.put("FileChooser.listForeground", Color.WHITE);

        UIManager.put("MenuBar.background", Color.DARK_GRAY); // Menu bar background color
        UIManager.put("MenuBar.border", BorderFactory.createLineBorder(Color.GRAY)); // Menu bar border

        UIManager.put("Menu.background", Color.DARK_GRAY); // Menu background color
        UIManager.put("Menu.foreground", Color.WHITE); // Menu text color
        UIManager.put("Menu.selectionBackground", Color.GRAY); // Highlight color when menu item is selected
        UIManager.put("Menu.selectionForeground", Color.WHITE); // Highlighted menu text color

        UIManager.put("MenuItem.background", Color.DARK_GRAY); // Menu item background color
        UIManager.put("MenuItem.foreground", Color.WHITE); // Menu item text color
        UIManager.put("MenuItem.selectionBackground", Color.GRAY); // Highlight color for selected menu item
        UIManager.put("MenuItem.selectionForeground", Color.WHITE); // Highlighted text color
        UIManager.put("MenuItem.border", BorderFactory.createEmptyBorder(5, 10, 5, 10)); // Padding for menu items

        UIManager.put("Panel.background", Color.DARK_GRAY);

        UIManager.put("Label.foreground", Color.WHITE);

        UIManager.put("Button.background", Color.GRAY); // Default background
        UIManager.put("Button.foreground", Color.WHITE); // Default text color
        UIManager.put("Button.focus", Color.DARK_GRAY); // Focus indicator color
        UIManager.put("Button.select", Color.LIGHT_GRAY); // Color when the button is pressed
        UIManager.put("Button.disabledText", Color.LIGHT_GRAY); // Text color when disabled
        UIManager.put("Button.disabledBackground", Color.DARK_GRAY); // Background color when disabled
        UIManager.put("Button.border", BorderFactory.createLineBorder(Color.WHITE)); // Optional border for buttons

        UIManager.put("ScrollPane.background", Color.DARK_GRAY);
        UIManager.put("ScrollPane.foreground", Color.WHITE); // Foreground color of the scroll pane

        UIManager.put("TitledBorder.titleColor", Color.WHITE);

        UIManager.put("Slider.background", Color.DARK_GRAY);
        UIManager.put("Slider.foreground", Color.WHITE);

        UIManager.put("TextField.background", Color.GRAY);
        UIManager.put("TextField.foreground", Color.WHITE);
        UIManager.put("TextField.caretForeground", Color.WHITE);

        UIManager.put("ComboBox.background", Color.DARK_GRAY);
        UIManager.put("ComboBox.foreground", Color.WHITE);
        UIManager.put("ComboBox.selectionBackground", Color.GRAY);
        UIManager.put("ComboBox.selectionForeground", Color.WHITE);
        UIManager.put("CheckBox.background", Color.DARK_GRAY);
        UIManager.put("CheckBox.foreground", Color.WHITE);

        UIManager.put("RadioButton.background", Color.DARK_GRAY);
        UIManager.put("RadioButton.foreground", Color.WHITE);

        UIManager.put("ProgressBar.background", Color.DARK_GRAY);
        UIManager.put("ProgressBar.foreground", Color.GRAY);
        UIManager.put("ProgressBar.selectionBackground", Color.DARK_GRAY);
        UIManager.put("ProgressBar.selectionForeground", Color.WHITE);

        UIManager.put("Spinner.background", Color.DARK_GRAY);
        UIManager.put("Spinner.foreground", Color.WHITE);

        UIManager.put("TextArea.background", Color.DARK_GRAY); // Background color for JTextArea
        UIManager.put("TextArea.foreground", Color.WHITE); // Text color for JTextArea
        UIManager.put("TextArea.caretForeground", Color.WHITE); // Caret (cursor) color
        UIManager.put("TextArea.selectionBackground", Color.GRAY); // Highlight color for selected text
        UIManager.put("TextArea.selectionForeground", Color.WHITE); // Highlighted text color
        UIManager.put("TextArea.border", BorderFactory.createLineBorder(Color.GRAY)); // Border for JTextArea

        UIManager.put("OptionPane.background", Color.DARK_GRAY); // Background of the dialog
        UIManager.put("OptionPane.foreground", Color.WHITE); // Text color for the dialog
        UIManager.put("OptionPane.messageForeground", Color.WHITE); // Foreground color for message text

        UIManager.put("Viewport.background", Color.DARK_GRAY); // Background color of the viewport (area containing the content)

        UIManager.put("ScrollBar.background", Color.DARK_GRAY); // Scroll bar background
        UIManager.put("ScrollBar.foreground", Color.WHITE); // Scroll bar color (thumb, etc.)
        UIManager.put("ScrollBar.thumbBackground", Color.GRAY); // Scroll bar thumb background (the draggable part)
        UIManager.put("ScrollBar.thumbForeground", Color.WHITE); // Scroll bar thumb text color (optional)
        UIManager.put("ScrollBar.trackBackground", Color.BLACK); // Track background (the part where the thumb slides)
        UIManager.put("ScrollBar.trackForeground", Color.DARK_GRAY); // Track foreground (optional for better visibility)
    }

    /**
     * Saves the configuration settings and reloads them from a configuration file.
     *
     * @param res Resolution value for the configuration.
     * @param epo Number of epochs for the configuration.
     * @param bat Batch size for the configuration.
     * @param lea Learning rate for the configuration.
     * @param pic Image path for the configuration.
     * @param ten Tensor model path for the configuration.
     */
    public void save_reload_config(int res, int epo, int bat, float lea, String pic, String ten) {
        System.out.println(res); // Output resolution for debugging
        String[][] values = { // Create values array for saving config
                {"img_path", pic}, // Image path
                {"ts_path", ten}, // Tensor model path
                {"resolution", String.valueOf(res)}, // Resolution
                {"batch", String.valueOf(bat)}, // Batch size
                {"epochs", String.valueOf(epo)}, // Epochs
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
                    {"batch", String.valueOf(batch_size)}, // Batch size
                    {"epochs", String.valueOf(epochs)}, // epochs
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

    public static class event_launch_CNN_playground implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            CNNPlayground gui = new CNNPlayground();
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set the window to hide on close
            gui.setTitle("CNN Playground"); // Set the window title
            gui.setVisible(true); // Make the window visible
            gui.setSize(new Dimension(1300, 1000));
            gui.setLocation(200, 50); // Position the window on the screen
        }
    }

    public static class event_set_params implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) { // Handle action event

            String[][] values = load_config(); // Load config values

            if (!(values[0][1].equals(prev_picture.getPath()) &&
                    values[1][1].equals(tensor_file.getPath()) &&
                    Integer.parseInt(values[2][1]) == resolution &&
                    Integer.parseInt(values[3][1]) == batch_size &&
                    Integer.parseInt(values[4][1]) == epochs &&
                    Float.parseFloat(values[5][1]) == learning_rate)) {

                setValues(values); // Update values due to new config detected
            }

            model_param gui = new model_param( // Create new model parameter settings window
                    (Main_UI) SwingUtilities.getWindowAncestor(detect_objects), // Pass the current instance of Main_UI
                    prev_picture.getPath(), // Pass previous picture path
                    tensor_file.getPath(), // Pass tensor model path
                    resolution, // Pass current resolution
                    batch_size, // Pass current batch size
                    epochs, // Pass current epochs
                    learning_rate // Pass current learning rate
            );

            gui.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Set close operation
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
                    {"batch", String.valueOf(batch_size)}, // Store batch size
                    {"epochs", String.valueOf(epochs)}, // Store epochs
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