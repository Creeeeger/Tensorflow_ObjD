package org.object_d;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

public class Main_UI extends JFrame {
    public JLabel label, img;
    public ImageIcon image = null;
    public JMenuBar menuBar;
    public JMenu file, model, database;
    public JMenuItem exit, load, load_database, reset_database, load_model, set_params, restore_last;
    public JScrollPane scrollPane;
    public JPanel leftPanel, rightPanel; // Panels for left and right boxes
    public File tensor_file;
    public JButton detect;

    public Main_UI() {
        setLayout(new FlowLayout()); // Use BorderLayout for main layout

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

        menuBar.add(file);
        menuBar.add(model);
        menuBar.add(database);

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

        scrollPane = new JScrollPane();
        rightPanel.add(scrollPane);

        detect = new JButton("Recognise Objects");
        leftPanel.add(detect);
        detect_ev d = new detect_ev();
        detect.addActionListener(d);
    }

    public static void main(String[] args) {
        Main_UI gui = new Main_UI();
        gui.setVisible(true);
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(600, 600);
        gui.setTitle("Object recognitions");

        HelloTensorFlow tensor = new HelloTensorFlow();
        System.out.println(tensor.version());
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

            config_handler.save_config(values);
            System.exit(0);
        }
    }

    public class detect_ev implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {

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
            getContentPane().add(scrollPane);

            // Revalidate and repaint the frame
            revalidate();
            repaint();
        }
    }


    public class event_restore_last implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            String[][] values = config_handler.load_config();
            for (String[] value : values) {
                System.out.println(value[0] + " " + value[1]);
            }

            // Create table model with data and column names
            DefaultTableModel model = new DefaultTableModel(values, new String[]{"Name", "Value"});
            JTable table1 = new JTable(model);

            // Create scroll pane and add table to it
            JScrollPane scrollPane = new JScrollPane(table1);

            // Add scroll pane to the frame
            getContentPane().add(scrollPane);

            // Revalidate and repaint the frame
            revalidate();
            repaint();
        }
    }

    public class event_load_tensor implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                tensor_file = fileChooser.getSelectedFile();
                System.out.println("Tensor file imported ");
            }
        }
    }

    public class event_set_params implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {

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

    public class event_load implements ActionListener {
        private final JLabel imageLabel;
        private final int desiredWidth = 400;
        private final int desiredHeight = 300;

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
                    Image scaledImage = originalImage.getScaledInstance(desiredWidth, desiredHeight, Image.SCALE_SMOOTH);
                    ImageIcon scaledIcon = new ImageIcon(scaledImage);
                    imageLabel.setIcon(scaledIcon);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }
    }
}