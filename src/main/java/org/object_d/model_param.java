package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class model_param extends JFrame {
    private final JLabel infos;
    private final JCheckBox setting1;
    private final JCheckBox setting2;
    private final JCheckBox setting3;
    private final JCheckBox setting4;
    private final JButton apply;

    public model_param(Main_UI mainUi) {
        setLayout(new BorderLayout(10, 10)); // Use BorderLayout with spacing

        JPanel settingsPanel = new JPanel();
        settingsPanel.setLayout(new BoxLayout(settingsPanel, BoxLayout.Y_AXIS));
        settingsPanel.setBorder(BorderFactory.createTitledBorder("Model Parameters")); // Add border with title

        infos = new JLabel("Select your settings and then press apply");
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

        apply = new JButton("Apply Settings");
        apply.setAlignmentX(Component.CENTER_ALIGNMENT); // Center the button
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add more space before button
        settingsPanel.add(apply);

        add(settingsPanel, BorderLayout.CENTER);

        // Add action listener for the apply button
        apply.addActionListener(new apply_event());
    }

    public class apply_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            boolean isSetting1Selected = setting1.isSelected();
            boolean isSetting2Selected = setting2.isSelected();
            boolean isSetting3Selected = setting3.isSelected();
            boolean isSetting4Selected = setting4.isSelected();
            //Apply settings to config link with config handler!!!


            setVisible(false);
        }
    }
}
//Link settings to config!!!
//Apply settings to detection