package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class reset_confirmation  extends JFrame {
    static JButton yes, no;

    public reset_confirmation(){
        setLayout(new GridLayout(1,1, 10, 10));
        BorderFactory.createTitledBorder("Decide to reset the Database");

        //init buttons
        yes = new JButton("Yes, reset Database");
        no = new JButton("No don't reset the Database");

        //add buttons
        add(yes);
        add(no);

        //add events
        no.addActionListener(new event_no());
        yes.addActionListener(new event_yes());
    }

    public class event_yes implements ActionListener{
        @Override
        public void actionPerformed(ActionEvent e) {
            database_handler.reset_init_db();
            setVisible(false);
        }
    }

    public class event_no implements ActionListener{
        @Override
        public void actionPerformed(ActionEvent e) {
            setVisible(false);

        }
    }
}
