package org.object_d;

import org.tensorflow.SavedModelBundle;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Objects;

import static org.object_d.config_handler.load_config;

public class Main_UI extends JFrame {
    public static JLabel label, img, image_path, model_path, result, output_img;
    public static JMenuBar menuBar;
    public static JMenu file, model, database, model_trainer, detector_menu;
    public static JMenuItem exit, load, load_database, reset_database, load_model, set_params, restore_last, train_model, self_detector, save_manually;
    public static JScrollPane scrollPane;
    public static JPanel leftPanel, rightPanel; // Panels for left and right boxes
    public static File tensor_file = new File("/");
    public static File picture = new File("/");
    public static JButton detect;
    public static int resolution, epochs, batch;
    public static float learning;
    static SavedModelBundle savedModelBundle;

    public Main_UI() {
        setLayout(new GridLayout(1, 2, 10, 10)); // Use horizontal grid layout with spacing

        // Create left and right panels
        leftPanel = new JPanel();
        leftPanel.setLayout(new BoxLayout(leftPanel, BoxLayout.Y_AXIS));
        leftPanel.setBorder(BorderFactory.createTitledBorder("Detection Panel")); // Add border with title

        rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Data Panel")); // Add border with title

        // Add panels to main frame
        add(leftPanel);
        add(rightPanel);

        // Create a dummy image placeholder
        BufferedImage placeholderImage = new BufferedImage(200, 200, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = placeholderImage.createGraphics();
        g2d.setColor(Color.GRAY);
        g2d.fillRect(0, 0, 200, 200);
        g2d.setColor(Color.BLACK);
        g2d.drawString("Image comes here", 50, 100);
        g2d.dispose();
        ImageIcon dummyImage = new ImageIcon(placeholderImage);

        // Left Panel Components
        label = new JLabel("Object detector");
        img = new JLabel(dummyImage);
        image_path = new JLabel("here comes the image path (select the actual image)");
        model_path = new JLabel("here comes the model path (select the folder with the tensor file in it)");
        result = new JLabel("Predicted results here");
        output_img = new JLabel(dummyImage);

        leftPanel.add(label);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(img);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(image_path);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(model_path);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(output_img);
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(result);

        detect = new JButton("Recognise Objects");
        detect.setEnabled(false);
        detect.addActionListener(new detect_ev());
        leftPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components
        leftPanel.add(detect);


        // Right Panel Components
        scrollPane = new JScrollPane();
        rightPanel.add(scrollPane);

        // Menu Bar Configuration
        menuBar = new JMenuBar();
        setJMenuBar(menuBar);

        file = new JMenu("File");
        load = new JMenuItem("Load image");
        load.addActionListener(new event_load(img));
        load_model = new JMenuItem("Load a tensor model");
        load_model.addActionListener(new event_load_tensor());
        restore_last = new JMenuItem("Restore last config");
        restore_last.addActionListener(new event_restore_last());
        save_manually = new JMenuItem("Save config");
        save_manually.addActionListener(new save_manu());
        exit = new JMenuItem("Save and Exit");
        exit.addActionListener(new event_exit());

        file.add(load);
        file.add(load_model);
        file.add(restore_last);
        file.add(save_manually);
        file.add(exit);

        menuBar.add(file);

        model = new JMenu("Model");
        set_params = new JMenuItem("Set model parameters");
        set_params.addActionListener(new event_set_params());
        model.add(set_params);
        menuBar.add(model);

        database = new JMenu("Database");
        load_database = new JMenuItem("Load database");
        load_database.addActionListener(new event_load_database());
        reset_database = new JMenuItem("Reset database");
        reset_database.addActionListener(new event_reset_database());
        database.add(load_database);
        database.add(reset_database);
        menuBar.add(database);

        model_trainer = new JMenu("Model creator");
        train_model = new JMenuItem("Train own models");
        train_model.addActionListener(new event_train());
        model_trainer.add(train_model);
        menuBar.add(model_trainer);

        detector_menu = new JMenu("Object detection v2");
        self_detector = new JMenuItem("detect objects with own models");
        self_detector.addActionListener(new create_detector_window());
        detector_menu.add(self_detector);
        menuBar.add(detector_menu);
    }

    public static void main(String[] args) {
        //Create config in case it doesn't exist
        File config = new File("config.xml");
        if (!config.exists()) {
            org.object_d.config_handler.create_config();
            System.out.println("Config Created");
        }

        //Create database in case it doesn't exist
        File database = new File("results.db");
        if (!database.exists()) {
            database_handler.CreateDatabase();
            System.out.println("Database created");
        }

        String[][] values_load = load_config();
        setValues(values_load);

        Main_UI gui = new Main_UI();
        gui.setVisible(true);
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(1200, 1000);
        gui.setTitle("Object Detector UI");
    }

    public static void setValues(String[][] values_load) {
        for (String[] value : Objects.requireNonNull(values_load)) {
            System.out.println(value[0] + " " + value[1]);
            switch (value[0]) {
                case "img_path":
                    File selectedFile = new File(value[1]);
                    try {
                        JLabel imageLabel = img;
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
                        if (ex.getClass() == NullPointerException.class) {
                            continue;
                        } else {
                            throw new RuntimeException(ex);
                        }
                    }
                    break;
                case "ts_path":
                    try {
                        tensor_file = new File(value[1]);
                        model_path.setText(tensor_file.getPath());
                        detect.setEnabled(true);
                        savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
                        System.out.println("Model loaded");
                    } catch (Exception ex) {
                        System.out.println("could not load tensor");
                    }
                    break;
                case "resolution":
                    resolution = Integer.parseInt(value[1]);
                    break;
                case "batch":
                    epochs = Integer.parseInt(value[1]);
                    break;
                case "epochs":
                    batch = Integer.parseInt(value[1]);
                    break;
                case "learning":
                    learning = Float.parseFloat(value[1]);
                    break;
                default:
                    System.out.println("Unknown setting: " + value[0]);
                    break;
            }
        }
    }

    public void save_reload_config(int res, int epo, int bat, float lea, String pic, String ten) {
        System.out.println(res);
        String[][] values = {
                {"img_path", pic},
                {"ts_path", ten},
                {"resolution", String.valueOf(res)},
                {"batch", String.valueOf(epo)},
                {"epochs", String.valueOf(bat)},
                {"learning", String.valueOf(lea)}
        };
        config_handler.save_config(values);

        String[][] values_load = load_config();
        setValues(values_load);
    }

    public static class save_manu implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            String[][] values = {
                    {"img_path", picture.getPath()},
                    {"ts_path", tensor_file.getPath()},
                    {"resolution", String.valueOf(resolution)},
                    {"batch", String.valueOf(epochs)},
                    {"epochs", String.valueOf(batch)},
                    {"learning", String.valueOf(learning)}
            };

            config_handler.save_config(values);
        }
    }

    public static class create_detector_window implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            trained_detector gui = new trained_detector();
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
            gui.setVisible(true);
            gui.setTitle("Object detector for own models");
            gui.setSize(600, 600);
            gui.setLocation(100, 100);
        }
    }

    public static class event_train implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            Trainer gui = new Trainer();
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
            gui.setVisible(true);
            gui.setSize(1400, 900);
            gui.setLocation(100, 100);
            gui.setTitle("Model trainer");
        }
    }

    public static class event_set_params implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            model_param gui = new model_param(picture.getPath(), tensor_file.getPath(), resolution, epochs, batch, learning);
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
            gui.setVisible(true);
            gui.setSize(1100, 550);
            gui.setLocation(100, 100);
            gui.setTitle("Model Parameter Settings");
        }
    }

    public static class event_exit implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            String[][] values = {
                    {"img_path", picture.getPath()},
                    {"ts_path", tensor_file.getPath()},
                    {"resolution", String.valueOf(resolution)},
                    {"batch", String.valueOf(epochs)},
                    {"epochs", String.valueOf(batch)},
                    {"learning", String.valueOf(learning)}
            };

            config_handler.save_config(values);

            System.exit(0);
        }
    }

    public static class detect_ev implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
            System.out.println("Model loaded");
            String[] result_array;
            result_array = org.tensorAction.detector.classify(image_path.getText(), savedModelBundle);

            File imagePath = new File(result_array[0]);
            ImageIcon icon = new ImageIcon(String.valueOf(imagePath));
            Image originalImage = icon.getImage();
            Image scaledImage = originalImage.getScaledInstance(400, 300, Image.SCALE_SMOOTH);
            ImageIcon scaledIcon = new ImageIcon(scaledImage);
            output_img.setIcon(scaledIcon);

            String[] data1D;
            String[][] data2d;

            try {
                data1D = result_array[1].split("\n");

                for (int i = 0; i < data1D.length; i++) {
                    data1D[i] = data1D[i].toLowerCase().substring(0, data1D[i].indexOf(":"));
                }
                data2d = new String[data1D.length][2];
                for (int i = 0; i < data1D.length; i++) {
                    data2d[i][0] = data1D[i];
                    data2d[i][1] = String.valueOf(1);
                }

                database_handler.addData(data2d);
            } catch (Exception exception) {
                System.out.println("Nothing got detected");
            }
            result.setText(result_array[1]);
        }
    }

    public static class event_load_tensor implements ActionListener { // returns tensor_file as loaded tensor

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
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
                throw new RuntimeException(ex);
            }
        }
    }

    public static class event_reset_database implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            database_handler.resetDatabase();
            database_handler.CreateDatabase();
            label.setText("Database reset");
        }
    }

    public static class event_load implements ActionListener {//returns picture as the loaded image
        JLabel imageLabel;

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
                    throw new RuntimeException(ex);
                }
            }
        }
    }

    public class event_restore_last implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            String[][] values = load_config();
            setValues(values);

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
}