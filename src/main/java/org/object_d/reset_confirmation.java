package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class reset_confirmation extends JFrame {
    // Static JButton variables for user confirmation buttons
    static JButton yes, no; // Buttons for user confirmation

    // Constructor for the reset confirmation dialog
    public reset_confirmation() {
        // Set the layout of the frame to a grid layout with 1 row and 1 column
        // The grid has horizontal and vertical gaps of 10 pixels
        setLayout(new GridLayout(1, 1, 10, 10));

        // Create a titled border for the dialog with a message
        BorderFactory.createTitledBorder("Decide to reset the Database");

        // Initialize the "Yes" button to allow the user to confirm resetting the database
        yes = new JButton("Yes, reset Database");
        // Initialize the "No" button to allow the user to cancel the reset action
        no = new JButton("No, don't reset the Database");

        // Add the "Yes" and "No" buttons to the frame
        add(yes);
        add(no);

        // Add action listeners to each button to handle user interactions
        no.addActionListener(new event_no()); // Listener for the "No" button
        yes.addActionListener(new event_yes()); // Listener for the "Yes" button
    }

    // Inner class for handling the action when the "Yes" button is clicked
    public class event_yes implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Call the method to reset the database when the user confirms
            database_handler.reset_init_db();
            // Hide the confirmation dialog after the action is completed
            setVisible(false);
            // Log the action to the console for debugging or record-keeping
            System.out.println("Reset the database");
            // Refresh the database view or user interface to show the changes
            database_utility.refresh();
        }
    }

    // Inner class for handling the action when the "No" button is clicked
    public class event_no implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Hide the confirmation dialog if the user cancels the reset action
            setVisible(false);
        }
    }
}