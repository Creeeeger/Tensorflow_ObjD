package org.object_d;

import javax.swing.*;
import java.awt.*;

public class Trainer extends JFrame {

    JLabel label;

    public Trainer(JFrame jFrame) {
        setLayout(new GridLayout(5,1));
        label = new JLabel("Model trainer");
        add(label);

    }
}
