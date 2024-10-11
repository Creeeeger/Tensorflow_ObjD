package org.object_d;

import ai.onnxruntime.OrtException;
import org.stabled.CLIApp;
import org.tensorAction.coreML_converter;
import org.tensorflow.SavedModelBundle;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Trainer extends JFrame {
    // Panel components for organizing the layout
    JPanel leftPanel; // Panel for the left section of the UI
    JPanel rightPanel; // Panel for the right section of the UI
    JPanel leftUpperPanel; // Upper panel of the left section
    JPanel leftLowerPanel; // Lower panel of the left section

    // Button components for user interactions
    JButton image_folder; // Button to select an image folder
    JButton stable_gen; // Button to trigger generation
    JButton output_path_button; // Button to specify the output path
    JButton create_model; // Button to create a model
    JButton sd4j; // Button for SD4J
    JButton CoreML_input_path; // Button for selecting CoreML input path
    JButton prepare; // Button to prepare data
    JButton model; // Button to manage model actions

    // Input components for user data entry
    JTextField command; // Text field for entering commands
    JSlider steps; // Slider to select the number of steps
    JSlider batch_size; // Slider to select batch size

    // Label components for displaying information
    JLabel images_path; // Label to show the selected images path
    JLabel gen; // Label for information
    JLabel output_path; // Label for displaying the output path
    JLabel ML_inp; // Label for displaying CoreML input information
    JLabel model_path; // Label for displaying the model path

    // File components for handling file paths
    File op_path_gen_img; // File for the generated image output path
    File img_for_train; // File for the images used in training
    File Ml_inp_file; // File for CoreML input
    File tensor_file; // File for tensor data storage

    // String variables for holding command strings and folder paths
    String command_string; // String to hold the command input
    String image_folder_String; // String to hold the image folder path

    // Model component for handling the saved model
    SavedModelBundle savedModelBundle; // Bundle to manage the saved model

    public Trainer() {
        // Set layout for the main frame with a horizontal grid layout
        setLayout(new GridLayout(1, 2, 10, 10)); // Use horizontal grid layout with spacing

        // Create left and right panels
        leftPanel = new JPanel(new GridLayout(2, 1)); // Panel for training controls
        leftPanel.setBorder(BorderFactory.createTitledBorder("Training Panel")); // Add border with title

        rightPanel = new JPanel(); // Panel for image generation controls
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Image Generation Panel")); // Add border with title

        // Create upper and lower panels for the left panel
        leftUpperPanel = new JPanel(); // Upper section for training model controls
        leftUpperPanel.setLayout(new BoxLayout(leftUpperPanel, BoxLayout.Y_AXIS));
        leftUpperPanel.setBorder(BorderFactory.createTitledBorder("Train Tensorflow model")); // Add border with title

        leftLowerPanel = new JPanel(); // Lower section for CoreML preparation controls
        leftLowerPanel.setLayout(new BoxLayout(leftLowerPanel, BoxLayout.Y_AXIS));
        leftLowerPanel.setBorder(BorderFactory.createTitledBorder("Prepare for CoreML training")); // Add border with title

        // Add upper and lower panels to the left panel
        leftPanel.add(leftUpperPanel);
        leftPanel.add(leftLowerPanel);

        // Add panels to the main frame
        add(leftPanel);
        add(rightPanel);

        // Left Upper Panel Components
        images_path = new JLabel("Select a folder with images first"); // Instruction label
        image_folder = new JButton("1. Select folder with images"); // Button for folder selection
        create_model = new JButton("2. Create model"); // Button to create a model
        create_model.setEnabled(false); // Initially disabled

        // Add components to upper panel
        leftUpperPanel.add(images_path);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftUpperPanel.add(image_folder);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftUpperPanel.add(create_model);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components

        // Left Lower Panel Components
        JLabel title = new JLabel("Prepare a folder of images with subfolders for training in CoreML."); // Instruction label
        ML_inp = new JLabel("Input path comes here"); // Placeholder for input path
        CoreML_input_path = new JButton("Select folder for input for conversion"); // Button for CoreML input folder
        model_path = new JLabel("Model path comes here"); // Placeholder for model path
        model = new JButton("Select tensor file"); // Button to select tensor file
        prepare = new JButton("Start preparing folder and JSON"); // Button to start preparation

        // Enable/disable buttons as necessary
        CoreML_input_path.setEnabled(true);
        model.setEnabled(false);
        prepare.setEnabled(false);

        // Add components to lower panel
        leftLowerPanel.add(title);
        leftLowerPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        leftLowerPanel.add(ML_inp);
        leftLowerPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        leftLowerPanel.add(CoreML_input_path);
        leftLowerPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        leftLowerPanel.add(model_path);
        leftLowerPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        leftLowerPanel.add(model);
        leftLowerPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        leftLowerPanel.add(prepare);
        leftLowerPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space

        // Right Panel Components
        gen = new JLabel("If you want to generate images with Stable Diffusion use this:"); // Instruction label
        command = new JTextField("1. Enter input for image generator", 75); // Text field for input
        output_path = new JLabel("Path of generated output images"); // Placeholder for output path
        output_path_button = new JButton("2. Select path for output generated images"); // Button for selecting output path
        JLabel step = new JLabel("3. Select steps for generation: more is better image quality"); // Instructions for steps
        steps = new JSlider(SwingConstants.HORIZONTAL, 1, 50, 5); // Slider for steps
        JLabel batch = new JLabel("4. Select how many images should get generated"); // Instructions for batch size
        batch_size = new JSlider(SwingConstants.HORIZONTAL, 1, 20, 1); // Slider for batch size
        stable_gen = new JButton("5. Generate images - Over web"); // Button to generate images via web
        sd4j = new JButton("5. Generate images directly"); // Button for direct image generation
        sd4j.setEnabled(false); // Initially disabled since pre steps need to be fulfilled

        // Add components to right panel
        rightPanel.add(gen);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space
        rightPanel.add(command);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space
        rightPanel.add(output_path);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space
        rightPanel.add(output_path_button);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        rightPanel.add(step);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        rightPanel.add(steps);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        rightPanel.add(batch);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        rightPanel.add(batch_size);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        rightPanel.add(stable_gen);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space
        rightPanel.add(sd4j);

        // Add action listeners to buttons
        image_folder.addActionListener(new image_path_function()); // Action for image folder button
        create_model.addActionListener(new create_model_event()); // Action for create model button
        output_path_button.addActionListener(new output_path_button_action()); // Action for output path button
        stable_gen.addActionListener(new stable_gen_event_web()); // Action for web generation
        sd4j.addActionListener(new sd4J_event()); // Action for direct generation

        CoreML_input_path.addActionListener(new Core_input()); // Action for CoreML input path button
        prepare.addActionListener(new convert_to_coreML()); // Action for preparation button
        model.addActionListener(new event_load_tensor()); // Action for tensor file button
    }

    public static boolean check_if_env_exists() {
        boolean does_exist = false; // Initialize a boolean to track if the environment exists
        Path path = Paths.get("stable_diff_env"); // Define the path to the environment directory

        try {
            // Check if the path exists and is a directory
            if (Files.exists(path) && Files.isDirectory(path)) {
                // Check if there are any files or directories within the specified path
                does_exist = Files.list(path).findAny().isPresent(); // Update does_exist if at least one item is found
            }
        } catch (IOException e) {
            // Handle any IO exceptions by throwing a RuntimeException
            throw new RuntimeException(e);
        }

        return does_exist; // Return the result indicating whether the environment exists or not
    }

    public void create_env() {
        try {
            // Update the label to inform the user that the environment setup is starting
            gen.setText("Setting up environment... Just wait");

            // Brew install dependencies
            // Create a process builder to run the brew command to install necessary dependencies
            ProcessBuilder brewProcessBuilder = new ProcessBuilder("brew", "install", "cmake", "protobuf", "rust", "python@3.10", "git", "wget");
            Process brewProcess = brewProcessBuilder.start(); // Start the brew process
            int brewExitCode = brewProcess.waitFor(); // Wait for the brew process to complete

            // Check if the brew command was successful
            if (brewExitCode != 0) {
                throw new RuntimeException("Brew command failed with exit code " + brewExitCode);
            }

            // Read and print the output of the brew command
            BufferedReader reader_brew = new BufferedReader(new InputStreamReader(brewProcess.getInputStream()));
            String line_brew;
            while ((line_brew = reader_brew.readLine()) != null) {
                System.out.println(line_brew); // Print each line of output from the brew command
            }

            // Clone the repository
            // Create a process builder to run the git clone command to clone the stable-diffusion-webui repository
            ProcessBuilder cloneProcessBuilder = new ProcessBuilder("git", "clone", "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git", "stable_diff_env");
            Process cloneProcess = cloneProcessBuilder.start(); // Start the git clone process

            // Read and print the output of the git clone command
            BufferedReader reader = new BufferedReader(new InputStreamReader(cloneProcess.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line); // Print each line of output from the git clone command
            }

            // Wait for the process to complete and check the exit value
            int exitCode = cloneProcess.waitFor(); // Wait for the clone process to complete
            if (exitCode != 0) {
                // Update the label to inform the user about the failure
                gen.setText("Rerun the Process now (Press generate images again)");
                throw new RuntimeException("Git clone failed with exit code " + exitCode);
            }

            // Update the label to inform the user of successful completion
            gen.setText("Git clone completed successfully and brew dependencies installed... Re-run generating images");
        } catch (IOException e) {
            // Handle IO exceptions
            System.err.println("IOException occurred: " + e.getMessage());
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            // Handle interruption exceptions
            System.err.println("InterruptedException occurred: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    public class event_load_tensor implements ActionListener { // ActionListener to load a tensor file

        @Override
        public void actionPerformed(ActionEvent e) {
            // Create a file chooser to select a directory
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Allow selection of directories only
            int returnValue = fileChooser.showOpenDialog(null); // Show the file chooser dialog

            // Check if the user approved the selection
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                // Get the selected file (directory)
                tensor_file = fileChooser.getSelectedFile();
                model_path.setText(tensor_file.getPath()); // Update the model path label with the selected path
                prepare.setEnabled(true); // Enable the prepare button
            }

            try {
                // Load the saved model bundle from the selected tensor file
                savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
                System.out.println("Model loaded"); // Print a confirmation message
            } catch (Exception ex) {
                // Handle any exceptions that occur during model loading
                throw new RuntimeException(ex); // throw RuntimeException
            }
        }
    }

    public class Core_input implements ActionListener { // ActionListener to handle input for CoreML

        @Override
        public void actionPerformed(ActionEvent e) {
            // Create a file chooser to select a directory
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set the file chooser to select directories
            int returnValue = fileChooser.showOpenDialog(null); // Show the file chooser dialog

            // Check if the user approved the selection
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                // Get the selected directory
                Ml_inp_file = fileChooser.getSelectedFile();
                ML_inp.setText(Ml_inp_file.getPath()); // Update the label with the selected path
                model.setEnabled(true); // Enable the model button
            }
        }
    }

    public class convert_to_coreML implements ActionListener { // ActionListener for converting to CoreML

        @Override
        public void actionPerformed(ActionEvent e) {
            // Initialize the CoreML converter and organize files for training
            coreML_converter.organiseFiles(Ml_inp_file.getPath()); // Organize files in the selected input directory
            coreML_converter.labelImages(tensor_file.getPath()); // Label images based on the loaded tensor file
        }
    }

    public class sd4J_event implements ActionListener { // ActionListener for generating images with sd4J

        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                // Generate images using the CLIApp with provided parameters
                CLIApp.gen(new String[]{}, steps.getValue(), command.getText(), batch_size.getValue(), op_path_gen_img.getPath());
            } catch (OrtException | IOException ex) {
                // Handle exceptions during image generation
                throw new RuntimeException(ex);
            }
        }
    }

    public class create_model_event implements ActionListener { // ActionListener for creating a model

        @Override
        public void actionPerformed(ActionEvent e) {
            // Get the path for training images
            image_folder_String = img_for_train.getPath(); // Store the path to images for model training
            System.out.println("Start Training Model"); // Indicate that training started
            try {
                // Access the TensorFlow model trainer with the image folder path
                org.tensorAction.tensorTrainerCNN.access(image_folder_String);
            } catch (IOException ex) {
                // Handle exceptions during model training
                throw new RuntimeException(ex);
            }
        }
    }

    public class stable_gen_event_web implements ActionListener { // ActionListener for generating images on the web

        @Override
        public void actionPerformed(ActionEvent e) {
            command_string = command.getText(); // Get the command input from the user
            // Provide instructions in case of download issues
            gen.setText("In case the download stops: download the safeTensor manually and put it into the model directory, in case the model gets stuck re run the sd web component");

            // Check if the environment exists
            if (check_if_env_exists()) {
                gen.setText("Starting generating images"); // Indicate that the image generation process is starting
                try {
                    // Create a new process builder for executing shell commands
                    ProcessBuilder processBuilder = new ProcessBuilder();
                    // Specify the command to start the shell (bash for Unix/Linux)
                    processBuilder.command("bash");
                    // Start the process
                    Process process = processBuilder.start();

                    // Create a PrintWriter to send commands to the shell
                    PrintWriter commandWriter = new PrintWriter(process.getOutputStream());

                    // Navigate to the stable_diff_env directory and run the webui.sh script
                    commandWriter.println("cd stable_diff_env");
                    commandWriter.println("./webui.sh"); // Execute the script to start the image generation
                    commandWriter.flush(); // Flush the commands to ensure they are executed

                    // Read the output of the command
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    // Print each line of the output to the console
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                } catch (Exception es) {
                    // Handle exceptions during process execution
                    throw new RuntimeException(es);
                }
            } else {
                // If the environment does not exist, create one
                create_env();
            }
        }
    }

    public class output_path_button_action implements ActionListener { // ActionListener for handling output path selection

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser(); // Create a file chooser
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set the file chooser to select directories
            int returnValue = fileChooser.showOpenDialog(null); // Show the dialog to choose a directory
            if (returnValue == JFileChooser.APPROVE_OPTION) { // Check if the user approved the selection
                op_path_gen_img = fileChooser.getSelectedFile(); // Get the selected directory
                output_path.setText(op_path_gen_img.getPath()); // Display the selected path in the output path label
                sd4j.setEnabled(true); // Enable the button for direct image generation
            }
        }
    }

    public class image_path_function implements ActionListener { // ActionListener for handling image folder selection

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser(); // Create a file chooser
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set the file chooser to select directories
            int returnValue = fileChooser.showOpenDialog(null); // Show the dialog to choose a directory
            if (returnValue == JFileChooser.APPROVE_OPTION) { // Check if the user approved the selection
                img_for_train = fileChooser.getSelectedFile(); // Get the selected directory
                images_path.setText(img_for_train.getPath()); // Display the selected path in the images path label
                create_model.setEnabled(true); // Enable the button for creating the model
            }
        }
    }
}