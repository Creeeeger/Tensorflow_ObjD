package org.object_d;

import org.tensorAction.detector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.*;
import java.util.ArrayList;
import java.util.UUID;

public class database_handler {

    public static void reset_init_db() {
        // initializing the database if it doesn't exist and resetting it if it does.

        Connection connection = null;
        Statement statement = null;

        try {
            // Register the SQLite JDBC driver so that we can use JDBC to interact with SQLite
            Class.forName("org.sqlite.JDBC");

            // Establish a connection to the database file named "results.db"
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Create a statement object for executing SQL statements
            statement = connection.createStatement();

            // File instance to check the existence of the database file
            File dbFile = new File("results.db");

            if (!dbFile.exists()) {
                // If the database file does not exist, create and initialize it
                System.out.println("Database doesn't exist - Create and initialise database");

                // SQL statement for creating the 'd_object' table with a primary key and data types
                String sql = "CREATE TABLE IF NOT EXISTS d_object (" +
                        "obj_id INTEGER PRIMARY KEY AUTOINCREMENT, " +   // Primary key with AUTOINCREMENT
                        "obj_name TEXT NOT NULL UNIQUE)";                // Object name must be unique

                // SQL statement for creating the 'link_obj' table with a composite primary key and foreign key constraints
                String sql1 = "CREATE TABLE IF NOT EXISTS link_obj (" +
                        "obj_id INTEGER NOT NULL, " +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "PRIMARY KEY (obj_id, date_id), " +                // Composite primary key using obj_id and date_id
                        "FOREIGN KEY (obj_id) REFERENCES d_object (obj_id))"; // Foreign key referring to d_object table

                // SQL statement for creating the 'obj_amt' table with foreign key constraints to 'link_obj'
                String sql2 = "CREATE TABLE IF NOT EXISTS obj_amt (" +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "amount INTEGER NOT NULL, " +
                        "date DATE NOT NULL, " +                           // Date of the record
                        "PRIMARY KEY (date_id), " +                        // Primary key on date_id
                        "FOREIGN KEY (date_id) REFERENCES link_obj (date_id))"; // Foreign key to link_obj table

                // Execute the SQL statements to create tables
                statement.executeUpdate(sql);
                statement.executeUpdate(sql1);
                statement.executeUpdate(sql2);

            } else {
                // If the database exists, we need to reset it by dropping the tables and re-initializing them
                System.out.println("Database exists - reset database");

                // Drop existing tables if they exist
                String dropTableSQL1 = "DROP TABLE IF EXISTS d_object";
                String dropTableSQL2 = "DROP TABLE IF EXISTS link_obj";
                String dropTableSQL3 = "DROP TABLE IF EXISTS obj_amt";

                // Execute drop statements to delete tables
                statement.executeUpdate(dropTableSQL1);
                statement.executeUpdate(dropTableSQL2);
                statement.executeUpdate(dropTableSQL3);

                System.out.println("Create and initialise database");

                // Re-create tables after dropping them

                // SQL statement for creating the 'd_object' table again
                String sql = "CREATE TABLE IF NOT EXISTS d_object (" +
                        "obj_id INTEGER PRIMARY KEY AUTOINCREMENT, " +   // Primary key with AUTOINCREMENT
                        "obj_name TEXT NOT NULL UNIQUE)";                // Object name must be unique

                // SQL statement for creating the 'link_obj' table with a composite primary key and foreign key constraints
                String sql1 = "CREATE TABLE IF NOT EXISTS link_obj (" +
                        "obj_id INTEGER NOT NULL, " +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "PRIMARY KEY (obj_id, date_id), " +              // Composite primary key using obj_id and date_id
                        "FOREIGN KEY (obj_id) REFERENCES d_object (obj_id))"; // Foreign key referring to d_object table

                // SQL statement for creating the 'obj_amt' table with foreign key constraints to 'link_obj'
                String sql2 = "CREATE TABLE IF NOT EXISTS obj_amt (" +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "amount INTEGER NOT NULL, " +
                        "date DATE NOT NULL, " +                         // Date of the record
                        "PRIMARY KEY (date_id), " +                      // Primary key on date_id
                        "FOREIGN KEY (date_id) REFERENCES link_obj (date_id))"; // Foreign key to link_obj table

                // Execute the SQL statements to create tables again
                statement.executeUpdate(sql);
                statement.executeUpdate(sql1);
                statement.executeUpdate(sql2);
            }

        } catch (Exception e) {
            // In case of an exception, throw a RuntimeException to indicate an unexpected error
            throw new RuntimeException(e);

        } finally {
            // Clean up resources to prevent resource leakage
            try {
                if (statement != null) {
                    statement.close(); // Close the SQL statement
                }
                if (connection != null) {
                    connection.close(); // Close the database connection
                }
            } catch (SQLException e) {
                System.out.println("Error occurred: " + e); // Print error if something goes wrong during cleanup
            }
        }
    }

    public static void addData(ArrayList<detector.entry> data) {
        // Method to add entries from an ArrayList into the database, updating or inserting as needed

        Connection connection = null;  // To store the connection to the database
        Statement statement = null;    // To execute SQL queries

        try {
            // Establish connection to the SQLite database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Loop through each entry in the provided ArrayList
            for (int i = 0; i < data.size(); i++) {
                String objName = data.get(i).getLabel().replace(" ", "");  // Get the label, remove spaces to normalize
                Date date = data.get(i).getDate();                         // Get the date associated with the data entry
                int dateId = UUID.randomUUID().hashCode();                 // Generate a unique identifier for the date

                // Create a new statement for executing SQL commands
                statement = connection.createStatement();

                // SQL query to check if the object already exists in the 'd_object' table
                String doesObjExist = "SELECT obj_id FROM d_object WHERE obj_name = '" + objName + "'";
                ResultSet objExist = statement.executeQuery(doesObjExist); // Execute the query

                int objId = -1;  // Variable to store the ID of the object, initialized to -1

                if (!objExist.next()) {
                    // If the object does not exist, insert a new record into 'd_object'
                    String insertOBJ = "INSERT INTO d_object (obj_name) VALUES ('" + objName + "')";
                    statement.executeUpdate(insertOBJ);

                    // Retrieve the obj_id of the newly inserted object
                    String getObjId = "SELECT obj_id FROM d_object WHERE obj_name = '" + objName + "'";
                    ResultSet newObj = statement.executeQuery(getObjId);
                    if (newObj.next()) {
                        objId = newObj.getInt("obj_id");  // Assign the new object ID
                    }
                } else {
                    // If the object already exists, get its obj_id
                    objId = objExist.getInt("obj_id");
                }

                // SQL query to check if the specific obj_id and date already exist in 'link_obj' and 'obj_amt' tables
                String doesDateExist = "SELECT link_obj.date_id " +
                        "FROM link_obj, obj_amt " + // Joins link_obj and obj_amt tables
                        "WHERE link_obj.obj_id = " + objId + " " + // Filter by object ID
                        "AND link_obj.date_id = obj_amt.date_id " + // Ensure matching date_id in both tables
                        "AND obj_amt.date = '" + date + "'"; // Filter by date in obj_amt table
                ResultSet dateExist = statement.executeQuery(doesDateExist);

                if (!dateExist.next()) {
                    // If there is no entry with the same object and date, add a new entry to 'link_obj'
                    String insertLinkObjSQL = "INSERT INTO link_obj (obj_id, date_id) " +
                            "VALUES (" + objId + ", " + dateId + ")";
                    statement.executeUpdate(insertLinkObjSQL);  // Insert the new link_obj entry

                    // Insert a new record into 'obj_amt' with an initial amount of 1 for the given date
                    String insertAmtSQL = "INSERT INTO obj_amt (date_id, amount, date) " +
                            "VALUES (" + dateId + ", 1, '" + date + "')";
                    statement.executeUpdate(insertAmtSQL);  // Insert the new obj_amt entry

                } else {
                    // If an entry with the same object and date exists, update the 'obj_amt' table to increment the amount
                    int existingDateId = dateExist.getInt("date_id");  // Get the existing date ID
                    String updateAmtSQL = "UPDATE obj_amt SET amount = amount + 1 WHERE date_id = " + existingDateId;
                    statement.executeUpdate(updateAmtSQL);  // Update the amount for the existing entry
                }
            }

            // Confirm that data has been added to the database
            System.out.println("Data added to the database");

        } catch (SQLException e) {
            // Handle SQL exceptions by wrapping them in a RuntimeException
            throw new RuntimeException(e);

        } finally {
            // Close the resources in the finally block to ensure proper cleanup
            try {
                if (statement != null)
                    statement.close();  // Close the statement
                if (connection != null)
                    connection.close();  // Close the database connection
            } catch (SQLException e) {
                // Handle any exceptions that may occur while closing resources
                System.out.println("Error occurred: " + e);
            }
        }
    }

    public static String[][] readDatabase() {
        // Method to read the database and return the data as a 2D String array.
        // This is intended for simple load actions, with more complex actions handled in other methods.

        class entries { // Define an inner class to store individual database entry data
            String name;  // Object name
            String date;  // Date associated with the object
            int amount;   // Amount value for the object on the specific date
        }

        ArrayList<entries> data = new ArrayList<>(); // Initialize an ArrayList to store entries from the database

        Connection connection;  // Database connection object
        Statement statement;    // Statement object for executing queries

        File dbfile = new File("results.db");  // File object representing the database file

        if (!dbfile.exists()) {
            // Check if the database file exists
            // If not, print a message and return null to indicate an issue
            System.out.println("Where is your DB gone?");
            return null;
        }

        try {
            // Establish a connection to the SQLite database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");
            statement = connection.createStatement(); // Create a statement object for executing SQL queries

            // SQL query to select data from multiple tables with specific conditions and sorting
            String sql = "SELECT d_object.obj_name, obj_amt.date, obj_amt.amount " +
                    "FROM d_object, link_obj, obj_amt " +
                    "WHERE d_object.obj_id = link_obj.obj_id " +  // Ensure the correct object is linked
                    "AND link_obj.date_id = obj_amt.date_id " +    // Match dates between link_obj and obj_amt
                    "ORDER BY obj_amt.date ASC, d_object.obj_name ASC;"; // Sort the results by date and then by object name

            ResultSet resultSet = statement.executeQuery(sql); // Execute the query and get the result set

            // Iterate through the result set and create entries to store the data
            while (resultSet.next()) {
                entries entry = new entries();  // Create a new entry object for each row of data
                entry.name = resultSet.getString("obj_name"); // Get the object name from the result set
                entry.date = resultSet.getString("date");     // Get the date value from the result set
                entry.amount = Integer.parseInt(resultSet.getString("amount")); // Get the amount, converting from String to int
                data.add(entry); // Add the entry to the ArrayList
            }

            // Convert the ArrayList into a 2D array of Strings for easier usage elsewhere
            // The array has three columns: name, date, and amount
            String[][] array_data = new String[data.size()][3];

            for (int i = 0; i < data.size(); i++) {
                entries entry = data.get(i);           // Get the entry from the ArrayList
                array_data[i][0] = entry.name;         // Column 1: Object name
                array_data[i][1] = entry.date;         // Column 2: Date
                array_data[i][2] = String.valueOf(entry.amount); // Column 3: Amount, converted to String
            }

            return array_data; // Return the 2D array with all the retrieved data

        } catch (Exception e) {
            // Catch any exceptions that may occur and rethrow them as RuntimeException
            // This simplifies error handling for the calling method
            throw new RuntimeException(e);
        }
    }

    public static void delete_entry(String name, String date, String amount) {
        // Method to delete a record based on provided object name, date, and amount.

        System.out.println("Delete record");

        Connection connection = null; // Connection object to manage the connection to the database
        Statement statement = null;   // Statement object to execute SQL commands

        try {
            // Register the JDBC driver for SQLite
            Class.forName("org.sqlite.JDBC");

            // Establish connection to the SQLite database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Create a statement for executing SQL queries
            statement = connection.createStatement();

            // Get the specific obj_id for the given obj_name
            String getObjIdQuery = "SELECT obj_id FROM d_object WHERE obj_name = '" + name + "'";
            ResultSet resultSet = statement.executeQuery(getObjIdQuery);

            int objId = -1;  // Initialize objId to an invalid value
            if (resultSet.next()) {
                // If a matching object is found, store its obj_id
                objId = resultSet.getInt("obj_id");
            }

            // Check if a valid obj_id was found
            if (objId == -1) {
                // If no obj_id was found for the provided name, print a message and exit the method
                System.out.println("No matching object found for name: " + name);
                return;
            }

            // Count how many records exist for this obj_id in the link_obj table
            String countQuery = "SELECT COUNT(*) FROM link_obj WHERE obj_id = " + objId;
            ResultSet countResultSet = statement.executeQuery(countQuery);

            int recordCount = 0;  // Initialize recordCount to store the count of linked records
            if (countResultSet.next()) {
                recordCount = countResultSet.getInt(1); // Get the count of records for this obj_id
            }

            // Delete based on the count of records for this obj_id
            if (recordCount == 1) {
                // If there's only one record, delete everything related to this obj_id

                // Delete from obj_amt using the date_id linked to this obj_id in the link_obj table
                String deleteObjAmt = "DELETE FROM obj_amt WHERE date_id IN (" +
                        "   SELECT date_id FROM link_obj WHERE obj_id = " + objId + ")";

                // Delete from link_obj using the obj_id
                String deleteLinkObj = "DELETE FROM link_obj WHERE obj_id = " + objId;

                // Delete from d_object using the obj_id
                String deleteDObject = "DELETE FROM d_object WHERE obj_id = " + objId;

                // Execute each delete statement in the correct order to avoid foreign key constraint violations
                statement.executeUpdate(deleteObjAmt);
                statement.executeUpdate(deleteLinkObj);
                statement.executeUpdate(deleteDObject);

                System.out.println("All records related to the object '" + name + "' were deleted successfully");

            } else if (recordCount > 1) {
                // If there are multiple records linked to this obj_id, only delete the specific record based on date and amount

                // Get the date_id associated with the obj_id and the provided date
                String selectDateId = "SELECT link_obj.date_id FROM link_obj " +
                        "JOIN obj_amt ON link_obj.date_id = obj_amt.date_id " + // Join to match link_obj with obj_amt
                        "WHERE link_obj.obj_id = " + objId + " AND obj_amt.date = '" + date + "'";

                ResultSet dateIdResultSet = statement.executeQuery(selectDateId);

                if (dateIdResultSet.next()) {
                    // If a matching date_id is found, store it
                    int dateId = dateIdResultSet.getInt("date_id");

                    // Delete from obj_amt for the specific date_id and amount
                    String deleteObjAmt = "DELETE FROM obj_amt WHERE date_id = " + dateId + " AND amount = " + Integer.parseInt(amount);
                    statement.executeUpdate(deleteObjAmt);

                    // Delete from link_obj for the specific obj_id and date_id
                    String deleteLinkObj = "DELETE FROM link_obj WHERE obj_id = " + objId + " AND date_id = " + dateId;
                    statement.executeUpdate(deleteLinkObj);

                    System.out.println("Specific record for date '" + date + "' and amount '" + amount + "' deleted successfully.");
                }
            }

        } catch (Exception e) {
            // Catch any exception and rethrow it as a RuntimeException to notify the caller of failure
            throw new RuntimeException(e);
        } finally {
            // Clean up resources
            try {
                if (statement != null) statement.close(); // Close Statement if it is open
                if (connection != null) connection.close(); // Close Connection if it is open
            } catch (SQLException ex) {
                // Print an error message if there's an issue closing the resources
                System.out.println("Error: " + ex);
            }
        }
    }

    public static String[][] searchData(String name, String date, String amount) {
        // Method for searching data in the database based on provided parameters

        // Declare Statement and Connection objects for database operations
        Statement statement;
        Connection connection;

        try {
            // Create an inner class to represent an entry with three data pieces
            class entities {
                String name_e;   // Name of the object
                String date_e;   // Date associated with the object
                int amount_e;    // Amount associated with the object
            }

            // Initialize an ArrayList to store the search results
            ArrayList<entities> data = new ArrayList<>();

            // Set up the connection to the SQLite database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");
            statement = connection.createStatement();

            // Create a StringBuilder to construct the SQL query
            StringBuilder sql = new StringBuilder();
            sql.append("SELECT d_object.obj_name, obj_amt.date, obj_amt.amount ")
                    .append("FROM d_object, link_obj, obj_amt ")
                    .append("WHERE d_object.obj_id = link_obj.obj_id ")
                    .append("AND link_obj.date_id = obj_amt.date_id ");

            // Add filters to the SQL query only if the corresponding parameters are provided
            if (name != null && !name.isEmpty()) {
                // Add a filter for the object's name using a LIKE clause with wildcards
                sql.append("AND d_object.obj_name LIKE '%").append(name).append("%' ");
            }

            if (date != null && !date.isEmpty()) {
                // Add a filter for the date using a LIKE clause with wildcards
                sql.append("AND obj_amt.date LIKE '%").append(date).append("%' ");
            }

            if (amount != null && !amount.isEmpty()) {
                // Add a filter for the amount using a LIKE clause with wildcards
                sql.append("AND obj_amt.amount LIKE '%").append(amount).append("%' ");
            }

            // Finalize the query by adding sorting criteria
            sql.append("ORDER BY obj_amt.date ASC, d_object.obj_name ASC;");

            // Execute the constructed SQL query and store the result in a ResultSet
            ResultSet resultSet = statement.executeQuery(sql.toString());

            // Iterate through the ResultSet and populate the ArrayList with entities
            while (resultSet.next()) {
                entities entry = new entities(); // Create a new entry for each result
                entry.name_e = resultSet.getString("obj_name"); // Get the object name
                entry.date_e = resultSet.getString("date");     // Get the date
                entry.amount_e = Integer.parseInt(resultSet.getString("amount")); // Get the amount
                data.add(entry); // Add the entry to the ArrayList
            }

            // Prepare a 2D array to store the final search results
            String[][] array_data = new String[data.size()][3];

            // Convert the ArrayList into a 2D array
            for (int i = 0; i < data.size(); i++) { // Loop through the ArrayList
                entities entry = data.get(i); // Get the current entry
                array_data[i][0] = entry.name_e;                   // Column 1: Name
                array_data[i][1] = entry.date_e;                   // Column 2: Date
                array_data[i][2] = String.valueOf(entry.amount_e); // Column 3: Amount
            }

            // Return the 2D array containing the search results
            return array_data;

        } catch (SQLException e) {
            // Catch any SQL exceptions and throw a RuntimeException
            throw new RuntimeException(e);
        }
    }

    public static void exportToCSV(String filePath) {
        // Method to export the database contents to a CSV file
        Connection connection = null; // Connection object for the database
        Statement statement = null;   // Statement object for executing SQL queries
        FileWriter csvWriter = null;  // FileWriter object for writing to the CSV file

        try {
            // Establish database connection
            Class.forName("org.sqlite.JDBC"); // Load the SQLite JDBC driver
            connection = DriverManager.getConnection("jdbc:sqlite:results.db"); // Connect to the SQLite database
            statement = connection.createStatement(); // Create a Statement object

            // Execute query to retrieve data from the database
            String query = "SELECT d_object.obj_name, obj_amt.date, obj_amt.amount " +
                    "FROM d_object, link_obj, obj_amt " +
                    "WHERE d_object.obj_id = link_obj.obj_id " +
                    "AND link_obj.date_id = obj_amt.date_id";

            ResultSet resultSet = statement.executeQuery(query); // Execute the query and store the result

            // Create CSV Writer
            csvWriter = new FileWriter(filePath); // Initialize FileWriter to create a new CSV file at the specified file path

            // Write CSV Header (assuming 3 columns: Name, Date, Amount)
            csvWriter.append("Name,Date,Amount\n"); // Write the header line to the CSV file

            // Write Data from ResultSet to CSV
            while (resultSet.next()) { // Loop through each row in the ResultSet
                String name = resultSet.getString("obj_name"); // Retrieve the object name
                String date = resultSet.getString("date");     // Retrieve the date
                String amount = resultSet.getString("amount"); // Retrieve the amount

                // Escape the data for CSV formatting
                csvWriter.append(escapeCSV(name)).append(",")  // Append the escaped name
                        .append(escapeCSV(date)).append(",")  // Append the escaped date
                        .append(escapeCSV(amount)).append("\n"); // Append the escaped amount and a newline
            }

            System.out.println("CSV file created successfully at: " + filePath); // Success message

        } catch (Exception e) {
            throw new RuntimeException(e); // Exception handling for any errors that occur

        } finally {
            try {
                // Close resources
                if (csvWriter != null) csvWriter.flush(); // Flush the FileWriter to ensure all data is written
                if (csvWriter != null) csvWriter.close(); // Close the FileWriter
                if (statement != null) statement.close(); // Close the Statement
                if (connection != null) connection.close(); // Close the Connection
            } catch (IOException | SQLException ex) {
                System.out.println("Error: " + ex); // Log any errors during resource cleanup
            }
        }
    }

    private static String escapeCSV(String data) {
        // Escape special characters for CSV (e.g., commas, quotes, newlines)
        if (data == null) {
            return "";  // Handle null values by returning an empty string
        }
        String escapedData = data; // Initialize escapedData with the original data

        // If data contains commas, quotes, or newlines, enclose it in double quotes
        if (data.contains(",") || data.contains("\"") || data.contains("\n")) {
            escapedData = "\"" + data.replace("\"", "\"\"") + "\""; // Replace quotes with escaped quotes and enclose in double quotes
        }
        return escapedData; // Return the escaped data
    }
}