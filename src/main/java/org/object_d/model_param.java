package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class model_param extends JFrame {
    private final JCheckBox setting1;
    private final JCheckBox setting2;
    private final JCheckBox setting3;
    private final JCheckBox setting4;
    private final String pic;  // Add instance variable for pic
    private final String ten;  // Add instance variable for ten

    public model_param(String pic, String ten) {
        setLayout(new BorderLayout(10, 10)); // Use BorderLayout with spacing

        this.pic = pic;  // Initialize pic
        this.ten = ten;  // Initialize ten

        JPanel settingsPanel = new JPanel();
        settingsPanel.setLayout(new BoxLayout(settingsPanel, BoxLayout.Y_AXIS));
        settingsPanel.setBorder(BorderFactory.createTitledBorder("Model Parameters")); // Add border with title

        JLabel infos = new JLabel("Select your settings and then press apply");
        settingsPanel.add(infos);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Add space between components

        setting1 = new JCheckBox("Setting 1", false);
        setting2 = new JCheckBox("Setting 2", false);
        setting3 = new JCheckBox("Setting 3", false);
        setting4 = new JCheckBox("Setting 4", false);

        settingsPanel.add(setting1);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Add space between components
        settingsPanel.add(setting2);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(setting3);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(setting4);

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
            boolean isSetting1Selected = setting1.isSelected();
            boolean isSetting2Selected = setting2.isSelected();
            boolean isSetting3Selected = setting3.isSelected();
            boolean isSetting4Selected = setting4.isSelected();
            Main_UI mainUI = new Main_UI();
            mainUI.save_reload_config(isSetting1Selected, isSetting2Selected, isSetting3Selected, isSetting4Selected, pic, ten);
            setVisible(false);
        }
    }
}
//Use right settings which we actually need!!!