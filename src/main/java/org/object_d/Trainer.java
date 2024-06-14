package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Trainer extends JFrame {
    JPanel leftPanel, rightPanel;
    JButton image_folder, stable_gen, output_path_button, create_model;
    JTextField command;
    JLabel images_path, gen, output_path;
    File op_path_gen_img, img_for_train;
    String command_string, output_gen_string, image_folder_String;

    public Trainer() {
        setLayout(new GridLayout(1, 2, 10, 10)); // Use horizontal grid layout with spacing

        // Create left and right panels
        leftPanel = new JPanel();
        leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS));
        leftPanel.setBorder(BorderFactory.createTitledBorder("Training Panel")); // Add border with title

        rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Image Generation Panel")); // Add border with title

        // Add panels to main frame
        add(leftPanel);
        add(rightPanel);

        // Left Panel Components
        images_path = new JLabel("Select a folder with images first");
        image_folder = new JButton("1. Select folder with images");
        create_model = new JButton("2. Create model");
        create_model.setEnabled(false);

        leftPanel.add(images_path);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(image_folder);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space between components
        leftPanel.add(create_model);

        // Right Panel Components
        gen = new JLabel("If you want to generate images with Stable Diffusion use this:");
        command = new JTextField("1. Enter input for image generator -- replace with your own request", 75);
        output_path = new JLabel("Path of generated output images");
        output_path_button = new JButton("2. Select path for output generated images");
        stable_gen = new JButton("3. Generate images");
        stable_gen.setEnabled(false);

        rightPanel.add(gen);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        rightPanel.add(command);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        rightPanel.add(output_path);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        rightPanel.add(output_path_button);
        rightPanel.add(Box.createRigidArea(new Dimension(0, 20)));
        rightPanel.add(stable_gen);

        // Add action listeners
        image_folder.addActionListener(new image_path_function());
        create_model.addActionListener(new create_model_event());
        output_path_button.addActionListener(new output_path_button_action());
        stable_gen.addActionListener(new stable_gen_event());
    }

    public void create_env() {
        try {
            //Brew install dependencies
            ProcessBuilder BrewProcessBuilder = new ProcessBuilder("brew install cmake protobuf rust python@3.10 git wget");
            Process BrewProcess = BrewProcessBuilder.start();
            int brewExitCode = BrewProcess.waitFor();
            if (brewExitCode != 0) {
                throw new RuntimeException("Brew command failed with exit code " + brewExitCode);
            }

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
                throw new RuntimeException("Git clone failed with exit code " + exitCode);
            }

            System.out.println("Git clone completed successfully and brew dependence install");
            gen.setText("Rerun the Process now (Press generate images again)");

        } catch (IOException e) {
            System.err.println("IOException occurred: " + e.getMessage());
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            System.err.println("InterruptedException occurred: " + e.getMessage());
            throw new RuntimeException(e);
        }
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

    public class create_model_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            image_folder_String = img_for_train.getPath(); //path for images
            System.out.println("Start Training Model");
            //create model logic!!!
        }
    }

    public class stable_gen_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            if (command.getText().isEmpty()) {
                command.setBackground(new Color(255, 1, 1));
            } else {
                command.setBackground(new Color(255, 255, 255));
            }

            if (op_path_gen_img == null) {
                output_path.setText("select the path first");
            }

            if (!command.getText().isEmpty() && !op_path_gen_img.getPath().isEmpty()) {
                command_string = command.getText();
                output_gen_string = op_path_gen_img.getPath();
                if (check_if_env_exists()) {
                    System.out.println("Starting generating images");
                    try {
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

                        // Wait for the process to exit
                        process.waitFor();

                    } catch (Exception es) {
                        es.printStackTrace();
                    }
                    //proceed with stable dif interaction!!!
                } else {
                    create_env();
                }
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
                stable_gen.setEnabled(true);
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

//proceed with stable dif interaction
//create model logic
//Rest of the setup process with git