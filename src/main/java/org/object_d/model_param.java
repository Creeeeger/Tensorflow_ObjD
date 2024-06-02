package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class model_param extends JFrame {
    JLabel infos;
    JCheckBox setting1, setting2, setting3, setting4;
    JButton apply;

    public model_param(Main_UI mainUi) {
        setLayout(new GridLayout(7, 1));

        infos = new JLabel("Select your settings and then press apply");
        add(infos);

        setting1 = new JCheckBox("Setting 1", false);
        add(setting1);

        setting2 = new JCheckBox("Setting 2", false);
        add(setting2);

        setting3 = new JCheckBox("Setting 3", false);
        add(setting3);

        setting4 = new JCheckBox("Setting 4", false);
        add(setting4);

        apply = new JButton("Apply Settings");
        add(apply);
        apply_event apply_event = new apply_event();
        apply.addActionListener(apply_event);
    }

    public class apply_event implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            //Apply settings to config!!!
            //Hide window!!!
        }
    }
}
//Link settings to config!!!
//Apply settings to detection