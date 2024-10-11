package org.object_d;

import org.tensorAction.detector;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

public class database_utility extends JFrame {
    // Static JLabel components for displaying various statuses and information
    static JLabel delete_status; // Label to show the status of delete operations
    static JLabel search_Info_name; // Label to display information about the searched name
    static JLabel search_info_date; // Label to display information about the searched date
    static JLabel search_info_amount; // Label to display information about the searched amount
    static JLabel delete_instruction; // Label for instructions related to deletion
    static JLabel selected_entry; // Label to show the currently selected entry
    static JLabel modify_hint; // Label that provide hints for modifying entries
    static JLabel del_info; // Label for displaying delete-related information

    // Static JTable components for displaying results
    static JTable result_table_left; // Table for displaying results on the left side
    static JTable result_table_middle; // Table for displaying results in the middle
    static JTable result_table_right; // Table for displaying results on the right side

    // Static DefaultTableModel components for managing table data
    static DefaultTableModel defaultTableModel; // Model for the left result table
    static DefaultTableModel middleTableModel; // Model for the middle result table
    static DefaultTableModel nonEditableModel; // Model for a non-editable table

    // Static Object array for initializing table data as a placeholder
    static Object[][] data; // Placeholder data for the tables

    // Static JPanel components for organizing the layout of the user interface
    static JPanel left_panel; // Panel for the left section of the UI
    static JPanel middle_panel; // Panel for the middle section of the UI
    static JPanel right_panel; // Panel for the right section of the UI

    // Static JButton components for performing actions within the UI
    static JButton reset_whole_db; // Button to reset the entire database
    static JButton delete_entry_button; // Button to trigger the deletion of an entry
    static JButton write_to_db; // Button to write data to the database
    static JButton csvExport; // Button to export data to a CSV file

    // Static JTextField components for user input
    static JTextField searchField_search; // Text field for entering search queries related to names
    static JTextField dateField_search; // Text field for entering search queries related to dates
    static JTextField amountField_search; // Text field for entering search queries related to amounts

    public database_utility() { // Constructor to create the design for the main window

        // Initial database access to fill tables during construction
        data = database_handler.readDatabase();

        // Set the main layout of the window to a grid layout with 1 row and 3 columns
        setLayout(new GridLayout(1, 3, 10, 10));
        BorderFactory.createTitledBorder("Database actions"); // Create a titled border for the main actions

        // Set up the left panel for search operations
        left_panel = new JPanel(); // Create the left panel
        left_panel.setLayout(new BoxLayout(left_panel, BoxLayout.Y_AXIS)); // Set vertical box layout
        left_panel.setBorder(BorderFactory.createTitledBorder("Search operation")); // Set a titled border

        // Set up the middle panel for modifying operations
        middle_panel = new JPanel(); // Create the middle panel
        middle_panel.setLayout(new BoxLayout(middle_panel, BoxLayout.Y_AXIS)); // Set vertical box layout
        middle_panel.setBorder(BorderFactory.createTitledBorder("Modify operation")); // Set a titled border

        // Set up the right panel for delete operations
        right_panel = new JPanel(); // Create the right panel
        right_panel.setLayout(new BoxLayout(right_panel, BoxLayout.Y_AXIS)); // Set vertical box layout
        right_panel.setBorder(BorderFactory.createTitledBorder("Delete operation")); // Set a titled border

        // Add all panels to the main window
        add(left_panel);
        add(middle_panel);
        add(right_panel);

        // Initialize the result table for the left panel with data and column names
        defaultTableModel = new DefaultTableModel(data, new Object[]{"Name", "Date", "Amount"});

        // Initialize the result table for the middle panel with the same data
        middleTableModel = new DefaultTableModel(data, new Object[]{"Name", "Date", "Amount"});

        // Create a non-editable table model for the right panel to prevent user edits
        nonEditableModel = new DefaultTableModel(data, new Object[]{"Name", "Date", "Amount"}) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;  // Disable editing for all cells
            }
        };

        // Create new JTable instances with the respective models
        result_table_left = new JTable(defaultTableModel);
        result_table_middle = new JTable(middleTableModel);
        result_table_right = new JTable(nonEditableModel); // This table is set to the non-editable model

        // Configure the right table's selection properties
        result_table_right.setEnabled(true);  // Allow row selection
        result_table_right.setRowSelectionAllowed(true); // Allow rows to be selected
        result_table_right.setColumnSelectionAllowed(false); // Disable column selection

        // Set up the search panel
        // Add a label and text field for searching by name
        search_Info_name = new JLabel("Search for specific objects by Name");
        searchField_search = new JTextField();
        searchField_search.setMaximumSize(new Dimension(800, 30)); // Set maximum size for the search field

        left_panel.add(search_Info_name); // Add name label to the panel
        left_panel.add(searchField_search); // Add name search field to the panel

        // Add a label and text field for searching by date
        search_info_date = new JLabel("Search for specific object by Date");
        dateField_search = new JTextField();
        dateField_search.setMaximumSize(new Dimension(800, 30)); // Set maximum size for the date field

        left_panel.add(search_info_date); // Add date label to the panel
        left_panel.add(dateField_search); // Add date search field to the panel

        // Add a label and text field for searching by amount
        search_info_amount = new JLabel("Search for specific objects by amount");
        amountField_search = new JTextField();
        amountField_search.setMaximumSize(new Dimension(800, 30)); // Set maximum size for the amount field

        left_panel.add(search_info_amount); // Add amount label to the panel
        left_panel.add(amountField_search); // Add amount search field to the panel

        // Add a label for displaying search results
        JLabel result = new JLabel("Here are the results displayed");
        left_panel.add(result);
        left_panel.add(result_table_left).setEnabled(false); // Disable the left table to prevent editing

        // Set up the modify panel
        // Add components for modifying entries
        modify_hint = new JLabel("Modify the entries you want to and then press write to database");
        middle_panel.add(modify_hint);
        middle_panel.add(result_table_middle); // Add the middle result table

        // Add hints related to modifying entries
        JLabel edit_hint = new JLabel("Un tick the field before saving");
        del_info = new JLabel("Save before using other tasks (data gets lost if not saved)");
        middle_panel.add(edit_hint); // Add edit hint label
        middle_panel.add(del_info); // Add save reminder label

        // Create and add a button for writing changes to the database
        write_to_db = new JButton("Write changes to database");
        middle_panel.add(write_to_db); // Add button to the middle panel

        // Setup the delete panel
        // Add status label for delete operations
        delete_status = new JLabel("Here appear updates on the delete status");
        delete_status.setForeground(Color.BLUE); // Set text color to blue for visibility
        right_panel.add(delete_status); // Add status label to the delete panel

        // Add a button to reset the entire database
        reset_whole_db = new JButton("Reset the whole database");
        right_panel.add(reset_whole_db); // Add reset button to the delete panel

        // Add the result table to the right panel
        right_panel.add(result_table_right); // Add non-editable table

        // Add components for deleting entries
        delete_instruction = new JLabel("Select entry to delete");
        selected_entry = new JLabel("Here will the entry appear you selected to delete");
        delete_entry_button = new JButton("Delete selected entry");
        delete_entry_button.setEnabled(false); // Disable delete button initially
        csvExport = new JButton("Export Database as CSV file"); // Button to export data

        // Add delete-related labels and buttons to the right panel
        right_panel.add(delete_instruction);
        right_panel.add(selected_entry);
        right_panel.add(delete_entry_button);
        right_panel.add(csvExport);

        // Actions section for handling user interactions
        // Left panel actions (search operations)
        searchField_search.getDocument().addDocumentListener(new event_change_search()); // Listen for changes in search field
        dateField_search.getDocument().addDocumentListener(new event_change_date()); // Listen for changes in date field
        amountField_search.getDocument().addDocumentListener(new event_change_amount()); // Listen for changes in amount field

        // Middle panel actions (modifying operations)
        write_to_db.addActionListener(new event_write_to_db()); // Listen for button clicks to write changes

        // Right panel actions (delete operations)
        reset_whole_db.addActionListener(new event_reset_database_UI()); // Listen for reset button clicks
        result_table_right.getSelectionModel().addListSelectionListener(new event_delete_select_raw()); // Listen for row selections in the right table
        delete_entry_button.addActionListener(new event_delete_entry()); // Listen for delete button clicks
        csvExport.addActionListener(new event_export_csv()); // Listen for CSV export button clicks
    }

    public static void searchOP(String name, String date, String amount) { // Method for searching data
        // Fetch the updated data from the database using the provided search criteria
        data = database_handler.searchData(name, date, amount);

        // Update the table models with the new data retrieved from the database
        defaultTableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"}); // Update the left result table
        nonEditableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"}); // Update the right result table

        // Revalidate and repaint the result tables to reflect the changes
        result_table_left.revalidate(); // Refresh the left table
        result_table_left.repaint();     // Repaint the left table to show updated data
        result_table_middle.revalidate(); // Refresh the middle table
        result_table_middle.repaint();     // Repaint the middle table to show updated data
        result_table_right.revalidate(); // Refresh the right table
        result_table_right.repaint();     // Repaint the right table to show updated data
    }

    public static void refresh() { // Method to refresh the entire dataset
        // Fetch the complete and updated data from the database
        data = database_handler.readDatabase();

        // Update all table models with the new data retrieved from the database
        defaultTableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"}); // Update the left result table
        middleTableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"}); // Update the middle result table
        nonEditableModel.setDataVector(data, new Object[]{"Name", "Date", "Amount"}); // Update the right result table

        // Revalidate and repaint the result tables to reflect the changes
        result_table_left.revalidate(); // Refresh the left table
        result_table_left.repaint();     // Repaint the left table to show updated data
        result_table_middle.revalidate(); // Refresh the middle table
        result_table_middle.repaint();     // Repaint the middle table to show updated data
        result_table_right.revalidate(); // Refresh the right table
        result_table_right.repaint();     // Repaint the right table to show updated data
    }

    // ActionListener for the reset database button
    public static class event_reset_database_UI implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Update the status label to indicate a reset confirmation is pending
            delete_status.setText("Wait for confirmation");

            // Create an instance of the reset confirmation dialog
            reset_confirmation gui = new reset_confirmation();
            gui.setVisible(true); // Make the dialog visible
            gui.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE); // Set close operation to hide on close
            gui.setLocation(100, 100); // Set the position of the dialog on the screen
            gui.setSize(500, 300); // Set the size of the dialog
        }
    }

    // ListSelectionListener for selecting a row in the right table (delete operations)
    public static class event_delete_select_raw implements ListSelectionListener {
        @Override
        public void valueChanged(ListSelectionEvent e) {
            // Check if the value change is final (not still adjusting)
            if (!e.getValueIsAdjusting()) {
                int selected_Row = result_table_right.getSelectedRow(); // Get the selected row index

                // Retrieve the data from each column of the selected row
                String name = result_table_right.getValueAt(selected_Row, 0).toString(); // Get the name from the first column
                String date = result_table_right.getValueAt(selected_Row, 1).toString(); // Get the date from the second column
                String amount = result_table_right.getValueAt(selected_Row, 2).toString(); // Get the amount from the third column

                // Concatenate the values into a single string (customize format as needed)
                String selectedRowContent = name + "\t" + date + "\t" + amount;

                // Set this concatenated string to the label or text field to display the selected entry
                selected_entry.setText(selectedRowContent);

                // Enable the delete button as an entry is selected
                delete_entry_button.setEnabled(true);
            }
        }
    }

    // DocumentListener for handling changes in the search fields
    public static class event_change_search implements DocumentListener {
        @Override
        public void insertUpdate(DocumentEvent e) {
            // Called when text is inserted into the search fields
            searchOP(searchField_search.getText(), dateField_search.getText(), amountField_search.getText());
        }

        @Override
        public void removeUpdate(DocumentEvent e) {
            // Called when text is removed from the search fields
            searchOP(searchField_search.getText(), dateField_search.getText(), amountField_search.getText());
        }

        @Override
        public void changedUpdate(DocumentEvent e) {
            // This method is not used but must be overridden as part of the DocumentListener interface
        }
    }

    // DocumentListener for handling changes in the date search field
    public static class event_change_date implements DocumentListener {
        @Override
        public void insertUpdate(DocumentEvent e) {
            // Called when text is inserted into the date search field
            searchOP(searchField_search.getText(), dateField_search.getText(), amountField_search.getText());
        }

        @Override
        public void removeUpdate(DocumentEvent e) {
            // Called when text is removed from the date search field
            searchOP(searchField_search.getText(), dateField_search.getText(), amountField_search.getText());
        }

        @Override
        public void changedUpdate(DocumentEvent e) {
            // This method is not used but must be overridden as part of the DocumentListener interface
        }
    }

    // DocumentListener for handling changes in the amount search field
    public static class event_change_amount implements DocumentListener {
        @Override
        public void insertUpdate(DocumentEvent e) {
            // Called when text is inserted into the amount search field
            searchOP(searchField_search.getText(), dateField_search.getText(), amountField_search.getText());
        }

        @Override
        public void removeUpdate(DocumentEvent e) {
            // Called when text is removed from the amount search field
            searchOP(searchField_search.getText(), dateField_search.getText(), amountField_search.getText());
        }

        @Override
        public void changedUpdate(DocumentEvent e) {
            // This method is not used but must be overridden as part of the DocumentListener interface
        }
    }

    // ActionListener for exporting data to a CSV file
    public static class event_export_csv implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Indicate the start of the export process
            System.out.println("Export started.");

            // Call the export method from the database handler
            database_handler.exportToCSV("exported_data.csv");

            // Indicate the completion of the export process
            System.out.println("Export done.");
        }
    }

    // ActionListener for deleting an entry from the database
    public static class event_delete_entry implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Log the entry to be deleted
            System.out.println("Entry to delete: " + selected_entry.getText());

            // Get the selected row index from the right table
            int selected_Row = result_table_right.getSelectedRow();

            // Call the delete method from the database handler with the selected entry's details
            database_handler.delete_entry(
                    result_table_right.getValueAt(selected_Row, 0).toString(), // Name
                    result_table_right.getValueAt(selected_Row, 1).toString(), // Date
                    result_table_right.getValueAt(selected_Row, 2).toString()  // Amount
            );

            // Refresh the UI to update the tables with the new data
            refresh();
        }
    }

    // ActionListener for writing modified data to the database
    public static class event_write_to_db implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println("Write modified data to Database");

            try {
                // Define the date format to match the expected input
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");

                // Get the table model from the middle result table
                TableModel model = result_table_middle.getModel();

                // Create a list to hold the detector.entry data for storage
                ArrayList<detector.entry> tab_res = new ArrayList<>();

                // Iterate over each row in the table model
                for (int row = 0; row < model.getRowCount(); row++) {
                    // Extract the label (Name) from the first column
                    String label = (String) model.getValueAt(row, 0);
                    // Extract the date string from the second column
                    String dateString = (String) model.getValueAt(row, 1);
                    // Extract the amount from the third column and convert it to an integer
                    int amount = Integer.parseInt(model.getValueAt(row, 2).toString());

                    java.sql.Date date;
                    try {
                        // Parse the date string into java.util.Date
                        java.util.Date utilDate = dateFormat.parse(dateString);
                        // Convert java.util.Date to java.sql.Date
                        date = new java.sql.Date(utilDate.getTime());
                    } catch (ParseException exc) {
                        // If there's an error in parsing the date, throw a runtime exception
                        throw new RuntimeException(exc);
                    }

                    // Add 'amount' number of entries to the list
                    for (int i = 0; i < amount; i++) {
                        detector.entry newEntry = new detector.entry();  // Create a new entry object
                        newEntry.setLabel(label);  // Set the label for the entry
                        newEntry.setDate(date);    // Set the date for the entry
                        tab_res.add(newEntry);      // Add the new entry to the list
                    }
                }

                // Log the entries created for verification
                for (detector.entry tabRe : tab_res) {
                    System.out.println(tabRe.getLabel() + " " + tabRe.getDate());
                }

                // Clear the existing database entries before adding new ones
                database_handler.reset_init_db();

                // Add the new entries from the list to the database
                database_handler.addData(tab_res);

                // Refresh the UI to update the tables with the new data
                refresh();

            } catch (Exception ee) {
                // If an exception occurs, display the error message in red
                del_info.setForeground(new Color(255, 0, 0));
                del_info.setText(ee.getClass() + " " + ee.getMessage());
                throw new RuntimeException(ee);
            }
        }
    }
}