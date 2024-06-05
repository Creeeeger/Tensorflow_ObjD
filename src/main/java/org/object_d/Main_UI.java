package org.object_d;

import org.tensorflow.SavedModelBundle;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.Objects;

public class Main_UI extends JFrame {
    public JLabel label, img, image_path, model_path, result;
    public ImageIcon image = null;
    public JMenuBar menuBar;
    public JMenu file, model, database, model_trainer;
    public JMenuItem exit, load, load_database, reset_database, load_model, set_params, restore_last, train_model;
    public JScrollPane scrollPane;
    public JPanel leftPanel, rightPanel; // Panels for left and right boxes
    public File tensor_file, picture;
    public JButton detect;
    SavedModelBundle savedModelBundle;

    public Main_UI() {
        setLayout(new GridLayout(2, 1)); // Use BorderLayout for main layout

        // Create left and right panels
        leftPanel = new JPanel();
        leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS)); // Vertical layout for left panel
        rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS)); // Vertical layout for right panel


        // Add panels to main frame
        add(leftPanel, BorderLayout.WEST);
        add(rightPanel, BorderLayout.EAST);
        label = new JLabel("Object detector");
        leftPanel.add(label);


        img = new JLabel(image);
        leftPanel.add(img);


        menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        file = new JMenu("File");
        model = new JMenu("Model");
        database = new JMenu("Database");
        model_trainer = new JMenu("Model creator");

        menuBar.add(file);
        menuBar.add(model);
        menuBar.add(database);
        menuBar.add(model_trainer);

        load_database = new JMenuItem("Load database");
        database.add(load_database);
        event_load_database eld = new event_load_database();
        load_database.addActionListener(eld);

        reset_database = new JMenuItem("Reset database");
        database.add(reset_database);
        event_reset_database erd = new event_reset_database();
        reset_database.addActionListener(erd);

        load_model = new JMenuItem("Load a tensor model");
        model.add(load_model);
        event_load_tensor elt = new event_load_tensor();
        load_model.addActionListener(elt);

        set_params = new JMenuItem("Set model parameters");
        model.add(set_params);
        event_set_params esp = new event_set_params();
        set_params.addActionListener(esp);

        restore_last = new JMenuItem("Restore last config");
        file.add(restore_last);
        event_restore_last erl = new event_restore_last();
        restore_last.addActionListener(erl);

        load = new JMenuItem("Load image");
        file.add(load);
        load.addActionListener(new event_load(img));

        exit = new JMenuItem("Save and Exit");
        file.add(exit);
        event_exit ex = new event_exit();
        exit.addActionListener(ex);

        train_model = new JMenuItem("Train own models");
        model_trainer.add(train_model);
        event_train tm = new event_train();
        train_model.addActionListener(tm);

        scrollPane = new JScrollPane();
        rightPanel.add(scrollPane);

        detect = new JButton("Recognise Objects");
        leftPanel.add(detect);
        detect.setEnabled(false);
        detect_ev d = new detect_ev();
        detect.addActionListener(d);

        image_path = new JLabel("here comes the image path");
        leftPanel.add(image_path);

        model_path = new JLabel("here comes the model path");
        leftPanel.add(model_path);

        result = new JLabel("Predicted results here");
        leftPanel.add(result);
    }

    public static void main(String[] args) {
        Main_UI gui = new Main_UI();
        gui.setVisible(true);
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(600, 600);
        gui.setTitle("Object recognitions");
    }

    public static class event_exit implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            //!! dummy values
            String[][] values = {
                    {"img_path", "/"},
                    {"version", "2.16.1"},
                    {"db_path", "/"},
                    {"set_date", "Thu May 02 10:38:25 BSssT 2024"}
            };
            //Use the actual variables instead of dummy data (event exit)!!!
            config_handler.save_config(values);
            System.exit(0);
        }
    }

    public class event_train implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            Trainer gui = new Trainer(Main_UI.this);
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
            gui.setVisible(true);
            gui.setSize(1000, 700);
            gui.setLocation(100, 100);
            gui.setTitle("Model trainer");
        }
    }

    public class detect_ev implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            //Add whole image detection logic (detect_ev image recogniser event)!!!
            //Create ui for outcome (detect_ev image recogniser event)!!!
        }
    }

    public class event_load_database implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            String[][] data = database_handler.readDatabase();

            // Create table model with data and column names
            DefaultTableModel model = new DefaultTableModel(data, new String[]{"Object", "Amount"});
            JTable table = new JTable(model);

            // Create scroll pane and add table to it
            JScrollPane scrollPane = new JScrollPane(table);

            // Add scroll pane to the frame
            rightPanel.add(scrollPane);

            // Revalidate and repaint the frame
            revalidate();
            repaint();
        }
    }


    public class event_restore_last implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            String[][] values = config_handler.load_config();
            for (String[] value : Objects.requireNonNull(values)) {
                System.out.println(value[0] + " " + value[1]);
                //Set the variables properly to the context they need (restore last)!!!
            }

            // Create table model with data and column names
            DefaultTableModel model = new DefaultTableModel(values, new String[]{"Name", "Value"});
            JTable table1 = new JTable(model);

            // Create scroll pane and add table to it
            JScrollPane scrollPane = new JScrollPane(table1);

            // Add scroll pane to the frame
            rightPanel.add(scrollPane);

            // Revalidate and repaint the frame
            revalidate();
            repaint();
            detect.setEnabled(true);
        }
    }

    public class event_load_tensor implements ActionListener { // returns tensor_file as loaded tensor

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                tensor_file = fileChooser.getSelectedFile();
                model_path.setText(tensor_file.getPath());
                detect.setEnabled(true);
            }

            try {
                savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
                System.out.println("Model loaded");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public class event_set_params implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            //Create param window!!!
            model_param gui = new model_param(Main_UI.this);
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
            gui.setVisible(true);
            gui.setSize(550, 550);
            gui.setLocation(100, 100);
            gui.setTitle("Model Parameter Settings");
        }
    }

    public class event_reset_database implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            database_handler.resetDatabase();
            database_handler.CreateDatabase();
            label.setText("Database reset");
        }
    }

    public class event_load implements ActionListener {//returns picture as the loaded image
        private final JLabel imageLabel;

        public event_load(JLabel imageLabel) {
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
                    picture = selectedFile;
                    image_path.setText(picture.getPath());
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }
    }
}
//Set the variables properly to the context they need (restore last)!!!
//Use the actual variables instead of dummy data (event exit)!!!
//Add whole image detection logic (detect_ev image recogniser event)!!!
//Reorganise Menu bar!!!