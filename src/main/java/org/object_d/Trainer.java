package org.object_d;

import ai.onnxruntime.OrtException;
import org.stabled.CLIApp;
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
    JPanel leftPanel, rightPanel, leftUpperPanel, leftLowerPanel;
    JButton image_folder, stable_gen, output_path_button, create_model, sd4j, CoreML_input_path, prepare, model;
    JTextField command;
    JSlider steps, batch_size;
    JLabel images_path, gen, output_path, ML_inp, model_path;
    File op_path_gen_img, img_for_train, Ml_inp_file, tensor_file;
    String command_string, output_gen_string, image_folder_String;
    SavedModelBundle savedModelBundle;

    public Trainer() {
        setLayout(new GridLayout(1, 2, 10, 10)); // Use horizontal grid layout with spacing

        // Create left and right panels
        leftPanel = new JPanel(new GridLayout(2, 1)); // Use GridLayout for equal spacing
        leftPanel.setBorder(BorderFactory.createTitledBorder("Training Panel")); // Add border with title

        rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Image Generation Panel")); // Add border with title

        // Create upper and lower panels for the left panel
        leftUpperPanel = new JPanel();
        leftUpperPanel.setLayout(new BoxLayout(leftUpperPanel, BoxLayout.Y_AXIS));
        leftUpperPanel.setBorder(BorderFactory.createTitledBorder("Train Tensorflow model")); // Add border with title

        leftLowerPanel = new JPanel();
        leftLowerPanel.setLayout(new BoxLayout(leftLowerPanel, BoxLayout.Y_AXIS));
        leftLowerPanel.setBorder(BorderFactory.createTitledBorder("Prepare for CoreML training")); // Add border with title

        // Add upper and lower panels to left panel
        leftPanel.add(leftUpperPanel);
        leftPanel.add(leftLowerPanel);

        // Add panels to main frame
        add(leftPanel);
        add(rightPanel);

        // Left Upper Panel Components
        images_path = new JLabel("Select a folder with images first");
        image_folder = new JButton("1. Select folder with images");
        create_model = new JButton("2. Create model");
        create_model.setEnabled(false);

        leftUpperPanel.add(images_path);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftUpperPanel.add(image_folder);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftUpperPanel.add(create_model);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components

        // Left Lower components
        JLabel title = new JLabel("Prepare a folder of images with subfolders for training in core ml Import the folder later to core ml and train it there");
        ML_inp = new JLabel("Input path comes here");
        CoreML_input_path = new JButton("Select folder for input for conversion");
        model_path = new JLabel("model path comes here");
        model = new JButton("select tensor file");
        prepare = new JButton("Start preparing folder and JSON");

        CoreML_input_path.setEnabled(true);
        model.setEnabled(false);
        prepare.setEnabled(false);

        leftLowerPanel.add(title);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftLowerPanel.add(ML_inp);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftLowerPanel.add(CoreML_input_path);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftLowerPanel.add(model_path);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftLowerPanel.add(model);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftLowerPanel.add(prepare);
        leftUpperPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components


        // Right Panel Components
        gen = new JLabel("If you want to generate images with Stable Diffusion use this:");
        command = new JTextField("1. Enter input for image generator", 75);
        output_path = new JLabel("Path of generated output images");
        output_path_button = new JButton("2. Select path for output generated images");
        JLabel step = new JLabel("3. Select steps for generation: more is better image quality");
        steps = new JSlider(SwingConstants.HORIZONTAL, 1, 50, 5);
        JLabel batch = new JLabel("4. Select how many images should get generated");
        batch_size = new JSlider(SwingConstants.HORIZONTAL, 1, 20, 1);
        stable_gen = new JButton("5. Generate images -  Over web");
        sd4j = new JButton("5. Generate images directly");
        sd4j.setEnabled(false);

        rightPanel.add(gen);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        rightPanel.add(command);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        rightPanel.add(output_path);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        rightPanel.add(output_path_button);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(step);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(steps);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(batch);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(batch_size);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(stable_gen);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(sd4j);

        // Add action listeners
        image_folder.addActionListener(new image_path_function());
        create_model.addActionListener(new create_model_event());
        output_path_button.addActionListener(new output_path_button_action());
        stable_gen.addActionListener(new stable_gen_event_web());
        sd4j.addActionListener(new sd4J_event());

        CoreML_input_path.addActionListener(new Core_input());
        prepare.addActionListener(new convert_to_coreML());
        model.addActionListener(new event_load_tensor());
    }

    public static boolean check_if_env_exists() {

        boolean does_exist = false;
        Path path = Paths.get("stable_diff_env");
        try {
            if (Files.exists(path) && Files.isDirectory(path)) {
                does_exist = Files.list(path).findAny().isPresent();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return does_exist;
    }

    public void create_env() {

        try {
            gen.setText("Setting up environment... Just wait");
            // Brew install dependencies
            ProcessBuilder brewProcessBuilder = new ProcessBuilder("brew", "install", "cmake", "protobuf", "rust", "python@3.10", "git", "wget");
            Process brewProcess = brewProcessBuilder.start();
            int brewExitCode = brewProcess.waitFor();
            if (brewExitCode != 0) {
                throw new RuntimeException("Brew command failed with exit code " + brewExitCode);
            }

            BufferedReader reader_brew = new BufferedReader(new InputStreamReader(brewProcess.getInputStream()));
            String line_brew;
            while ((line_brew = reader_brew.readLine()) != null) {
                System.out.println(line_brew);
            }

            // Clone the repository
            ProcessBuilder cloneProcessBuilder = new ProcessBuilder("git", "clone", "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git", "stable_diff_env");
            Process cloneProcess = cloneProcessBuilder.start();

            // Read and print the output of the git clone command
            BufferedReader reader = new BufferedReader(new InputStreamReader(cloneProcess.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            // Wait for the process to complete and check the exit value
            int exitCode = cloneProcess.waitFor();
            if (exitCode != 0) {
                gen.setText("Rerun the Process now (Press generate images again)");
                throw new RuntimeException("Git clone failed with exit code " + exitCode);
            }

            gen.setText("Git clone completed successfully and brew dependencies installed... Re-run generating images");
        } catch (IOException e) {
            System.err.println("IOException occurred: " + e.getMessage());
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            System.err.println("InterruptedException occurred: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    public class event_load_tensor implements ActionListener { // returns tensor_file as loaded tensor

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                tensor_file = fileChooser.getSelectedFile();
                model_path.setText(tensor_file.getPath());
                prepare.setEnabled(true);
            }

            try {
                savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
                System.out.println("Model loaded");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public class Core_input implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set the file chooser to select directories
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                Ml_inp_file = fileChooser.getSelectedFile();
                ML_inp.setText(Ml_inp_file.getPath());
                model.setEnabled(true);
            }
        }
    }

    public class convert_to_coreML implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            //initialise detector
            org.tensorAction.coreML_converter.organiseFiles(Ml_inp_file.getPath());
            org.tensorAction.coreML_converter.labelImages(tensor_file.getPath());
        }
    }

    public class sd4J_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                CLIApp.gen(new String[]{}, steps.getValue(), command.getText(), batch_size.getValue(), op_path_gen_img.getPath());
            } catch (OrtException | IOException ex) {
                throw new RuntimeException(ex);
            }
        }
    }

    public class create_model_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            image_folder_String = img_for_train.getPath(); //path for images
            System.out.println("Start Training Model");
            try {
                org.tensorAction.tensorTrainer.access(image_folder_String);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        }
    }

    public class stable_gen_event_web implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            command_string = command.getText();
            if (check_if_env_exists()) {
                gen.setText("Starting generating images");
                try {
                    gen.setText("Wait some time it could take long :/");
                    // Create a new process builder
                    ProcessBuilder processBuilder = new ProcessBuilder();

                    // Specify the command to start the shell (bash for Unix/Linux, cmd for Windows)
                    processBuilder.command("bash");

                    // Start the process
                    Process process = processBuilder.start();

                    // Create a PrintWriter to send commands to the shell
                    PrintWriter commandWriter = new PrintWriter(process.getOutputStream());

                    // Send a command to the shell
                    commandWriter.println("cd stable_diff_env");
                    commandWriter.println("./webui.sh");
                    commandWriter.flush();

                    // Get the output of the command
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                } catch (Exception es) {
                    es.printStackTrace();
                }
            } else {
                create_env();
            }
        }
    }

    public class output_path_button_action implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set the file chooser to select directories
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                op_path_gen_img = fileChooser.getSelectedFile();
                output_path.setText(op_path_gen_img.getPath());
                sd4j.setEnabled(true);
            }
        }
    }

    public class image_path_function implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY); // Set the file chooser to select directories
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                img_for_train = fileChooser.getSelectedFile();
                images_path.setText(img_for_train.getPath());
                create_model.setEnabled(true);
            }
        }
    }
}