package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

public class Trainer extends JFrame {

    JLabel label, images_path, output_path;
    JButton image_folder, stable_gen, output_path_button;
    JTextField command;
    File op_path_gen_img, img_for_train;
    String command_string, output_gen_string, image_folder_String;

    public Trainer(JFrame jFrame) {
        setLayout(new GridLayout(5, 1));
        label = new JLabel("Model trainer");
        add(label);

        images_path = new JLabel("Select a folder with images first");
        add(images_path);

        image_folder = new JButton("Select folder with images");
        add(image_folder);
        image_path_function image_path_function = new image_path_function();
        image_folder.addActionListener(image_path_function);

        JLabel gen = new JLabel("If you want to generate images with Stable Diffusion use this");
        add(gen);

        JLabel command_hint = new JLabel("Enter input for image generator");
        add(command_hint);

        command = new JTextField(40);
        add(command);

        output_path = new JLabel("Path of generated output images");
        add(output_path);

        output_path_button = new JButton("Select path for images");
        add(output_path_button);
        oppbe oppbe = new oppbe();
        output_path_button.addActionListener(oppbe);

        stable_gen = new JButton("generate images");
        add(stable_gen);
        stable_gen_event stable_gen_event = new stable_gen_event();
        stable_gen.addActionListener(stable_gen_event);
    }

    public class stable_gen_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            if (command.getText().isEmpty()) {
                command.setBackground(new Color(255, 1, 1));
            } else if (op_path_gen_img.getPath().isEmpty()) {
                output_path.setBackground(new Color(255, 0, 0));
            } else {
                command_string = command.getText();
                output_gen_string = op_path_gen_img.getPath();
                //Stable diff interface for generation!!
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
                image_folder_String = img_for_train.getPath();
                System.out.println(image_folder_String);
            }
        }
    }
}