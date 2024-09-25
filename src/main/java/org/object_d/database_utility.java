package org.object_d;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class database_utility extends JFrame {
    static JLabel delete_status, search_Info_name, search_info_date, search_info_amount, delete_instruction, selected_entry, modify_hint;
    static JTable result_table_left, result_table_middle, result_table_right;
    static DefaultTableModel defaultTableModel, nonEditableModel;
    static Object[][] data; //Init the data for the tables as a placeholder
    static JPanel left_panel, middle_panel, right_panel; //Initialize all the panels for the different tasks
    static JButton reset_whole_db, delete_entry_button, write_to_db;
    static JTextField searchField_search, dateField_search, amountField_search;

    public database_utility() { //Create Design for Window

        data = database_handler.readDatabase();

        //Set main layout
        setLayout(new GridLayout(1, 3, 10, 10));
        BorderFactory.createTitledBorder("Database actions"); //Create sub title

        //setup left panel
        left_panel = new JPanel(); // Setup search panel
        left_panel.setLayout(new BoxLayout(left_panel, BoxLayout.Y_AXIS));
        left_panel.setBorder(BorderFactory.createTitledBorder("Search operation"));

        //setup middle panel
        middle_panel = new JPanel(); //Setup modifier panel
        middle_panel.setLayout(new BoxLayout(middle_panel, BoxLayout.Y_AXIS));
        middle_panel.setBorder(BorderFactory.createTitledBorder("Modify operation"));

        //Setup right panel
        right_panel = new JPanel(); //Setup Delete panel
        right_panel.setLayout(new BoxLayout(right_panel, BoxLayout.Y_AXIS));
        right_panel.setBorder(BorderFactory.createTitledBorder("Delete operation"));

        //Add all panels to view window
        add(left_panel);
        add(middle_panel);
        add(right_panel);

        //initialise the result table for left and middle
        defaultTableModel = new DefaultTableModel(data, new Object[]{"Name", "Date", "Amount"});

        // Create a non-editable table model for the right panel
        nonEditableModel = new DefaultTableModel(data, new Object[]{"Name", "Date", "Amount"}) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;  // Disable editing for all cells
            }
        };

        //Create the new tables
        result_table_left = new JTable(defaultTableModel);
        result_table_middle = new JTable(defaultTableModel);
        result_table_right = new JTable(nonEditableModel); //set to new model to disable editing

        //Since we don't want the user to be able to edit the right table we need to give it special permissions
        result_table_right.setEnabled(true);  // Allow row selection
        result_table_right.setRowSelectionAllowed(true); // Allow rows to be selected
        result_table_right.setColumnSelectionAllowed(false); // Disable column selection

        //search panel
        //Add the name search to the panel
        search_Info_name = new JLabel("Search for specific objects by Name");
        searchField_search = new JTextField();
        searchField_search.setMaximumSize(new Dimension(800, 30));

        left_panel.add(search_Info_name);
        left_panel.add(searchField_search);

        //Add the date search to panel
        search_info_date = new JLabel("Search for specific object by Date");
        dateField_search = new JTextField();
        dateField_search.setMaximumSize(new Dimension(800, 30));

        left_panel.add(search_info_date);
        left_panel.add(dateField_search);

        //Add the search by amount to the panel
        search_info_amount = new JLabel("Search for specific objects by amount");
        amountField_search = new JTextField();
        amountField_search.setMaximumSize(new Dimension(800, 30));

        left_panel.add(search_info_amount);
        left_panel.add(amountField_search);

        //output hint is added
        JLabel result = new JLabel("Here are the results displayed");
        left_panel.add(result);
        left_panel.add(result_table_left).setEnabled(false);

        //modify panel
        //add the components
        modify_hint = new JLabel("Modify the entries you want to and then press write to database");
        middle_panel.add(modify_hint);
        middle_panel.add(result_table_middle);
        write_to_db = new JButton("Write changes to database");
        middle_panel.add(write_to_db);

        //delete panel
        //add the status label to the delete section
        delete_status = new JLabel("Here comes updates on delete states");
        delete_status.setForeground(Color.BLUE);
        right_panel.add(delete_status);

        //add total reset button and initialize it
        reset_whole_db = new JButton("Reset the whole database");
        right_panel.add(reset_whole_db);

        //add the table there
        right_panel.add(result_table_right);

        //add the delete hint, entry hint and delete button
        delete_instruction = new JLabel("Select entry to delete");
        selected_entry = new JLabel("Here will the entry appear you selected to delete");
        delete_entry_button = new JButton("Delete selected entry");
        delete_entry_button.setEnabled(false);

        right_panel.add(delete_instruction);
        right_panel.add(selected_entry);
        right_panel.add(delete_entry_button);

        //Actions section
        //left panel actions
        searchField_search.getDocument().addDocumentListener(new event_change_search());
        dateField_search.getDocument().addDocumentListener(new event_change_date());
        amountField_search.getDocument().addDocumentListener(new event_change_amount());

        //Middle panel actions
        write_to_db.addActionListener(new event_write_to_db());

        //right panel actions
        reset_whole_db.addActionListener(new event_reset_database_UI());
        result_table_right.getSelectionModel().addListSelectionListener(new event_delete_select_raw());
        delete_entry_button.addActionListener(new event_delete_entry());
    }

    public static void main(String[] args) {
        database_utility gui = new database_utility();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(1400, 800);
        gui.setTitle("Database utility");
        gui.setVisible(true);
        gui.setLocation(100, 100);
    }

    public static class event_reset_database_UI implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            delete_status.setText("Wait for confirmation");
            reset_confirmation gui = new reset_confirmation();
            gui.setVisible(true);
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
            gui.setLocation(100, 100);
            gui.setSize(500, 300);
        }
    }

    public static class event_delete_select_raw implements ListSelectionListener {

        @Override
        public void valueChanged(ListSelectionEvent e) {
            if (!e.getValueIsAdjusting()) {
                int selected_Row = result_table_right.getSelectedRow();

                // Retrieve the data from each column of the selected row
                String name = result_table_right.getValueAt(selected_Row, 0).toString();
                String date = result_table_right.getValueAt(selected_Row, 1).toString();
                String amount = result_table_right.getValueAt(selected_Row, 2).toString();

                // Concatenate the values into a single string (customize format as needed)
                String selectedRowContent = name + "\t" + date + "\t" + amount;

                // Set this concatenated string to the label or text field
                selected_entry.setText(selectedRowContent);

                delete_entry_button.setEnabled(true);
            }
        }
    }

    public class event_delete_entry implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println("Entry to delete: " + selected_entry.getText());
            int selected_Row = result_table_right.getSelectedRow();
            database_handler.delete_entry(
                    result_table_right.getValueAt(selected_Row, 0).toString(),
                    result_table_right.getValueAt(selected_Row, 1).toString(),
                    result_table_right.getValueAt(selected_Row, 2).toString()
            );
            refresh();
        }
    }
    public void refresh() {
        // Fetch the updated data from the database
        data = database_handler.readDatabase();

        // Update the table models with the new data
        defaultTableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"});
        nonEditableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"});

        // Revalidate and repaint the result tables and left panel
        result_table_left.revalidate();
        result_table_left.repaint();
        result_table_middle.revalidate();
        result_table_middle.repaint();
        result_table_right.revalidate();
        result_table_right.repaint();
        left_panel.revalidate();
        left_panel.repaint();
    }

    public static class event_write_to_db implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println("Write modified data to DB");
        }
    }

    public static class event_change_search implements DocumentListener {

        @Override
        public void insertUpdate(DocumentEvent e) {

        }

        @Override
        public void removeUpdate(DocumentEvent e) {

        }

        @Override
        public void changedUpdate(DocumentEvent e) {

        }
    }

    public static class event_change_date implements DocumentListener {

        @Override
        public void insertUpdate(DocumentEvent e) {

        }

        @Override
        public void removeUpdate(DocumentEvent e) {

        }

        @Override
        public void changedUpdate(DocumentEvent e) {

        }
    }

    public static class event_change_amount implements DocumentListener {

        @Override
        public void insertUpdate(DocumentEvent e) {

        }

        @Override
        public void removeUpdate(DocumentEvent e) {

        }

        @Override
        public void changedUpdate(DocumentEvent e) {

        }
    }
}

/*
Todo
event_write_to_db
event_change_search
event_change_date
event_change_amount
*/