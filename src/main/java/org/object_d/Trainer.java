package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Trainer extends JFrame {

    public JPanel leftPanel, rightPanel; // Panels for left and right boxes
    JLabel images_path, output_path;
    JButton image_folder, stable_gen, output_path_button, create_model;
    JTextField command;
    File op_path_gen_img, img_for_train;
    String command_string, output_gen_string, image_folder_String;


    public Trainer(JFrame jFrame) {
        setLayout(new GridLayout(2, 1));

        // Create left and right panels
        leftPanel = new JPanel();
        leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS)); // Vertical layout for left panel
        rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS)); // Vertical layout for right panel

        // Add panels to main frame
        add(leftPanel, BorderLayout.NORTH);
        add(rightPanel, BorderLayout.SOUTH);


        images_path = new JLabel("Select a folder with images first");
        leftPanel.add(images_path);

        image_folder = new JButton("1. Select folder with images");
        leftPanel.add(image_folder);
        image_path_function image_path_function = new image_path_function();
        image_folder.addActionListener(image_path_function);

        create_model = new JButton("2. Create model");
        create_model.setEnabled(false);
        leftPanel.add(create_model);
        create_model_event create_model_event = new create_model_event();
        create_model.addActionListener(create_model_event);


        JLabel gen = new JLabel("If you want to generate images with Stable Diffusion use this");
        rightPanel.add(gen);

        command = new JTextField("1. Enter input for image generator -- replace with your own request", 75);
        rightPanel.add(command);

        output_path = new JLabel("Path of generated output images");
        rightPanel.add(output_path);

        output_path_button = new JButton("2. Select path for output generated images");
        rightPanel.add(output_path_button);
        oppbe oppbe = new oppbe();
        output_path_button.addActionListener(oppbe);

        stable_gen = new JButton("3. Generate images");
        stable_gen.setEnabled(false);
        rightPanel.add(stable_gen);
        stable_gen_event stable_gen_event = new stable_gen_event();
        stable_gen.addActionListener(stable_gen_event);
    }

    public static void create_env() {
        try {
            Path path = Paths.get("stable_diff_env");
            Files.createDirectory(path);
            //Rest of the setup process with git!!!
        } catch (IOException e) {
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

    public static void main(String[] args) {
        //dummy for single function testing
    }

    public class create_model_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            image_folder_String = img_for_train.getPath(); //path for images
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
                    //proceed with stable dif interaction!!!
                } else {
                    create_env();
                }
            }
        }
    }

    public class oppbe implements ActionListener {

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

//Make Ui nicer
//proceed with stable dif interaction
//create model logic
//Rest of the setup process with git