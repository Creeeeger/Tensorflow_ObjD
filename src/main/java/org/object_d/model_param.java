package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class model_param extends JFrame {
    int res, epo, bat;
    float lea;
    JTextField resolution, epochs, batch, display_scale;
    JLabel resolution_desc, epochs_desc, batch_desc, display_scale_desc, infos;
    String pic, ten;

    public model_param(String pic, String ten, int res, int epo, int bat, float lea) {
        setLayout(new BorderLayout(10, 10)); // Use BorderLayout with spacing

        this.pic = pic;  // Initialize pic
        this.ten = ten;  // Initialize ten
        this.res = res;  // and the rest
        this.epo = epo;
        this.bat = bat;
        this.lea = lea;

        JPanel settingsPanel = new JPanel();
        settingsPanel.setLayout(new BoxLayout(settingsPanel, BoxLayout.Y_AXIS));
        settingsPanel.setBorder(BorderFactory.createTitledBorder("Model Parameters for training models")); // Add border with title

        infos = new JLabel("Select your settings and then press apply");
        settingsPanel.add(infos);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components

        resolution_desc = new JLabel("Picture size - to x * x pixel the images will be downscaled first before processing (larger value more information but longer processing time)");
        resolution = new JTextField(String.valueOf(res), 4);

        epochs_desc = new JLabel("Epochs - How many training rounds");
        epochs = new JTextField(String.valueOf(epo), 4);

        batch_desc = new JLabel("Batch size - how many images should be used for training at once");
        batch = new JTextField(String.valueOf(bat), 4);

        display_scale_desc = new JLabel("Data visualisation scale for the displaying and analysis of training");
        display_scale = new JTextField(String.valueOf(lea), 10);

        // Add space between components
        settingsPanel.add(resolution_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(resolution);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(epochs_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(epochs);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(batch_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(batch);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(display_scale_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(display_scale);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        JButton apply = new JButton("Apply Settings");
        apply.setAlignmentX(Component.CENTER_ALIGNMENT); // Center the button
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space before button
        settingsPanel.add(apply);

        add(settingsPanel, BorderLayout.CENTER);
        apply.addActionListener(new apply_event());
    }

    public class apply_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                int res = Integer.parseInt(resolution.getText());
                int epo = Integer.parseInt(epochs.getText());
                int bat = Integer.parseInt(batch.getText());
                float lea = Float.parseFloat(display_scale.getText());
                Main_UI mainUI = new Main_UI();
                mainUI.save_reload_config(res, epo, bat, lea, pic, ten);
                setVisible(false);

            } catch (Exception x) {
                infos.setForeground(Color.RED);
                infos.setText("Wrong input in the text fields");
                System.out.println("Wrong input in the text fields");
            }
        }
    }
}