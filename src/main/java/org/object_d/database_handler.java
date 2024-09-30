package org.object_d;

import org.tensorAction.detector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.*;
import java.util.ArrayList;
import java.util.UUID;

public class database_handler {

    public static void reset_init_db() { //new initialization method and reset method (2 in 1)
        Connection connection = null;
        Statement statement = null;

        try {
            //register the driver
            Class.forName("org.sqlite.JDBC");

            // Establish connection to the database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Create a statement
            statement = connection.createStatement();

            File dbFile = new File("results.db"); //check for db file

            if (!dbFile.exists()) { //if not existing create new db and init it
                System.out.println("Database doesn't exist - Create and initialise database");

                // Create the `d_object` table with a proper primary key and data types
                String sql = "CREATE TABLE IF NOT EXISTS d_object (" +
                        "obj_id INTEGER PRIMARY KEY AUTOINCREMENT, " +   // Primary key with AUTOINCREMENT
                        "obj_name TEXT NOT NULL UNIQUE)";                // Unique object name

                // Create the `link_obj` table with a composite primary key and foreign key constraints
                String sql1 = "CREATE TABLE IF NOT EXISTS link_obj (" +
                        "obj_id INTEGER NOT NULL, " +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "PRIMARY KEY (obj_id, date_id), " +            // Composite primary key
                        "FOREIGN KEY (obj_id) REFERENCES d_object (obj_id))"; // Foreign key constraint

                // Create the `obj_amt` table with a foreign key constraint to `link_obj`
                String sql2 = "CREATE TABLE IF NOT EXISTS obj_amt (" +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "amount INTEGER NOT NULL, " +
                        "date DATE NOT NULL, " +                        // Move `date` here from `link_obj`
                        "PRIMARY KEY (date_id), " +                     // Primary key on date_id
                        "FOREIGN KEY (date_id) REFERENCES link_obj (date_id))"; // Foreign key constraint

                // Execute the SQL statements
                statement.executeUpdate(sql);
                statement.executeUpdate(sql1);
                statement.executeUpdate(sql2);

            } else { //in case the file exists drop tables and re init them
                System.out.println("Database exists - reset database");

                // Drop the tables if they exists
                String dropTableSQL1 = "DROP TABLE IF EXISTS d_object";
                String dropTableSQL2 = "DROP TABLE IF EXISTS link_obj";
                String dropTableSQL3 = "DROP TABLE IF EXISTS obj_amt";

                // Execute each drop statement individually
                statement.executeUpdate(dropTableSQL1);
                statement.executeUpdate(dropTableSQL2);
                statement.executeUpdate(dropTableSQL3);

                System.out.println("Create and initialise database");

                // Create the `d_object` table with a proper primary key and data types
                String sql = "CREATE TABLE IF NOT EXISTS d_object (" +
                        "obj_id INTEGER PRIMARY KEY AUTOINCREMENT, " +   // Primary key with AUTOINCREMENT
                        "obj_name TEXT NOT NULL UNIQUE)";                // Unique object name

                // Create the `link_obj` table with a composite primary key and foreign key constraints
                String sql1 = "CREATE TABLE IF NOT EXISTS link_obj (" +
                        "obj_id INTEGER NOT NULL, " +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "PRIMARY KEY (obj_id, date_id), " +            // Composite primary key
                        "FOREIGN KEY (obj_id) REFERENCES d_object (obj_id))"; // Foreign key constraint

                // Create the `obj_amt` table with a foreign key constraint to `link_obj`
                String sql2 = "CREATE TABLE IF NOT EXISTS obj_amt (" +
                        "date_id INTEGER NOT NULL UNIQUE, " +
                        "amount INTEGER NOT NULL, " +
                        "date DATE NOT NULL, " +                        // Move `date` here from `link_obj`
                        "PRIMARY KEY (date_id), " +                     // Primary key on date_id
                        "FOREIGN KEY (date_id) REFERENCES link_obj (date_id))"; // Foreign key constraint

                // Execute the SQL statements
                statement.executeUpdate(sql);
                statement.executeUpdate(sql1);
                statement.executeUpdate(sql2);
            }

        } catch (Exception e) {
            throw new RuntimeException(e); //for random goofy exceptions we throw a RtE

        } finally {
            // Close resources
            try {
                if (statement != null)
                    statement.close();
                if (connection != null)
                    connection.close();

            } catch (SQLException e) {
                System.out.println("Error occurred: " + e);
            }
        }
    }

    public static void addData(ArrayList<detector.entry> data) {
        Connection connection = null;
        Statement statement = null;

        try {
            // Establish connection to the database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            for (int i = 0; i < data.size(); i++) {
                String objName = data.get(i).getLabel().replace(" ", "");
                Date date = data.get(i).getDate();
                int dateId = UUID.randomUUID().hashCode();  // Generate a random unique ID for each date entry

                // Create a statement
                statement = connection.createStatement();

                // Query to check if the object already exists in the d_object table
                String doesObjExist = "SELECT obj_id FROM d_object WHERE obj_name = '" + objName + "'";
                ResultSet objExist = statement.executeQuery(doesObjExist);

                int objId = -1;  // Initialize objId to store the object id

                if (!objExist.next()) {
                    // If the object doesn't exist, insert it into d_object
                    String insertOBJ = "INSERT INTO d_object (obj_name) VALUES ('" + objName + "')";
                    statement.executeUpdate(insertOBJ);

                    // Retrieve the obj_id of the newly inserted object
                    String getObjId = "SELECT obj_id FROM d_object WHERE obj_name = '" + objName + "'";
                    ResultSet newObj = statement.executeQuery(getObjId);
                    if (newObj.next()) {
                        objId = newObj.getInt("obj_id");
                    }
                } else {
                    // If the object exists, get its obj_id
                    objId = objExist.getInt("obj_id");
                }

                // Now check if this obj_id and date already exist in the link_obj table
                String doesDateExist = "SELECT link_obj.date_id " +
                        "FROM link_obj, obj_amt " + // Added space at the end of the line
                        "WHERE link_obj.obj_id = " + objId + " " + // Added spaces around 'objId' and after the clause
                        "AND link_obj.date_id = obj_amt.date_id " + // Added spaces around 'link_obj.date_id'
                        "AND obj_amt.date = '" + date + "'"; // Added space before AND
                ResultSet dateExist = statement.executeQuery(doesDateExist);

                if (!dateExist.next()) {
                    // If the same object but different date, insert a new entry into link_obj table
                    String insertLinkObjSQL = "INSERT INTO link_obj (obj_id, date_id) " +
                            "VALUES (" + objId + ", " + dateId + ")";
                    statement.executeUpdate(insertLinkObjSQL);

                    // Set the amount to 1 for the new date entry in obj_amt table
                    String insertAmtSQL = "INSERT INTO obj_amt (date_id, amount, date) " +
                            "VALUES (" + dateId + ", 1, '" + date + "')";
                    statement.executeUpdate(insertAmtSQL);

                } else {
                    // If the same object and same date, increment the amount in obj_amt table
                    int existingDateId = dateExist.getInt("date_id");  // Get the existing date_id
                    String updateAmtSQL = "UPDATE obj_amt SET amount = amount + 1 WHERE date_id = " + existingDateId;
                    statement.executeUpdate(updateAmtSQL);
                }
            }

            System.out.println("Data added to the database");

        } catch (SQLException e) {
            throw new RuntimeException(e);

        } finally {
            // Close resources
            try {
                if (statement != null)
                    statement.close();  // Close Statement
                if (connection != null)
                    connection.close();  // Close Connection
            } catch (SQLException e) {
                System.out.println("Error occurred: " + e);
            }
        }
    }

    public static String[][] readDatabase() { //Read the database for the simple load up action. For complex actions refer to the other methods

        class entries { //create the entry class with our 3 dataPieces
            String name;
            String date;
            int amount;
        }

        ArrayList<entries> data = new ArrayList<>(); // init the array list

        Connection connection;
        Statement statement;

        File dbfile = new File("results.db");

        if (!dbfile.exists()) { //issue handling in case db got lost Tbh I don't know how this could happen but nvm lets make it issue proof
            System.out.println("Where is your DB gone?");
            return null;
        }

        try {
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");
            statement = connection.createStatement();

            String sql = "SELECT d_object.obj_name, obj_amt.date, obj_amt.amount " +
                    "FROM d_object, link_obj, obj_amt " +
                    "WHERE d_object.obj_id = link_obj.obj_id " +
                    "AND link_obj.date_id = obj_amt.date_id " +
                    "ORDER BY obj_amt.date ASC, d_object.obj_name ASC;";

            ResultSet resultSet = statement.executeQuery(sql);

            while (resultSet.next()) {
                entries entry = new entries();
                entry.name = resultSet.getString("obj_name");
                entry.date = resultSet.getString("date");
                entry.amount = Integer.parseInt(resultSet.getString("amount"));
                data.add(entry);
            }

            // Now the array should have 3 columns (name, date, and amount)
            String[][] array_data = new String[data.size()][3];

            for (int i = 0; i < data.size(); i++) {
                entries entry = data.get(i);
                array_data[i][0] = entry.name;                   // Column 1: Name
                array_data[i][1] = entry.date;                   // Column 2: Date
                array_data[i][2] = String.valueOf(entry.amount); // Column 3: Amount
            }

            return array_data;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void delete_entry(String name, String date, String amount) {
        System.out.println("Delete record");

        Connection connection = null;
        Statement statement = null;

        try {
            // Register the driver
            Class.forName("org.sqlite.JDBC");

            // Establish connection to the database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Create a statement
            statement = connection.createStatement();

            // Step 1: Get the specific obj_id for the given obj_name
            String getObjIdQuery = "SELECT obj_id FROM d_object WHERE obj_name = '" + name + "'";

            ResultSet resultSet = statement.executeQuery(getObjIdQuery);
            int objId = -1;
            if (resultSet.next()) {
                objId = resultSet.getInt("obj_id");
            }

            // Check if a valid obj_id was found
            if (objId == -1) {
                System.out.println("No matching object found for name: " + name);
                return;
            }

            // Step 2: Count how many records exist for this obj_id
            String countQuery = "SELECT COUNT(*) FROM link_obj WHERE obj_id = " + objId;
            ResultSet countResultSet = statement.executeQuery(countQuery);
            int recordCount = 0;
            if (countResultSet.next()) {
                recordCount = countResultSet.getInt(1);
            }

            // Step 3: Delete based on the count of records for this obj_id
            if (recordCount == 1) {
                // If there's only one record, delete everything (d_object, link_obj, and obj_amt for that obj_id)

                // Delete from obj_amt (use the date_id from link_obj)
                String deleteObjAmt = "DELETE FROM obj_amt WHERE date_id IN (" +
                        "   SELECT date_id FROM link_obj WHERE obj_id = " + objId + ")";

                // Delete from link_obj
                String deleteLinkObj = "DELETE FROM link_obj WHERE obj_id = " + objId;

                // Delete from d_object
                String deleteDObject = "DELETE FROM d_object WHERE obj_id = " + objId;

                // Execute each query in the appropriate order (to avoid foreign key constraint errors)
                statement.executeUpdate(deleteObjAmt);
                statement.executeUpdate(deleteLinkObj);
                statement.executeUpdate(deleteDObject);

                System.out.println("All records related to the object '" + name + "' were deleted successfully");

            } else if (recordCount > 1) {
                // If there are multiple records for this obj_id, only delete the specific date and amount

                // Get the date_id associated with this obj_id and date
                String selectDateId = "SELECT link_obj.date_id FROM link_obj " +
                        "JOIN obj_amt ON link_obj.date_id = obj_amt.date_id " +
                        "WHERE link_obj.obj_id = " + objId + " AND obj_amt.date = '" + date + "'";
                ResultSet dateIdResultSet = statement.executeQuery(selectDateId);

                if (dateIdResultSet.next()) {
                    int dateId = dateIdResultSet.getInt("date_id");

                    // Delete from obj_amt for the specific date_id
                    String deleteObjAmt = "DELETE FROM obj_amt WHERE date_id = " + dateId + " AND amount = " + Integer.parseInt(amount);
                    statement.executeUpdate(deleteObjAmt);

                    // Delete from link_obj for the specific date_id
                    String deleteLinkObj = "DELETE FROM link_obj WHERE obj_id = " + objId + " AND date_id = " + dateId;
                    statement.executeUpdate(deleteLinkObj);

                    System.out.println("Specific record for date '" + date + "' and amount '" + amount + "' deleted successfully.");
                }
            }

        } catch (Exception e) {
            throw new RuntimeException(e); // Exception handling
        } finally {
            // Clean up resources
            try {
                if (statement != null) statement.close();
                if (connection != null) connection.close();
            } catch (SQLException ex) {
                System.out.println("Error: " + ex);
            }
        }
    }

    public static String[][] searchData(String name, String date, String amount) {
        //method for searching data
        //setup connection and statement
        Statement statement;
        Connection connection;

        try {
            //Create entity class
            class entities {
                String name_e;
                String date_e;
                int amount_e;
            }

            ArrayList<entities> data = new ArrayList<>();

            //setup connection with driver
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");
            statement = connection.createStatement();

            StringBuilder sql = new StringBuilder();
            sql.append("SELECT d_object.obj_name, obj_amt.date, obj_amt.amount ")
                    .append("FROM d_object, link_obj, obj_amt ")
                    .append("WHERE d_object.obj_id = link_obj.obj_id ")
                    .append("AND link_obj.date_id = obj_amt.date_id ");

            // Add filters only if the variables are provided
            if (name != null && !name.isEmpty()) {
                sql.append("AND d_object.obj_name LIKE '%").append(name).append("%' "); // Wildcard before and after name
            }

            if (date != null && !date.isEmpty()) {
                sql.append("AND obj_amt.date LIKE '%").append(date).append("%' "); // Wildcard before and after date
            }

            if (amount != null && !amount.isEmpty()) {
                sql.append("AND obj_amt.amount LIKE '%").append(amount).append("%' "); // Wildcard for amount if needed
            }

            // Finalize the query with sorting
            sql.append("ORDER BY obj_amt.date ASC, d_object.obj_name ASC;");

            // Execute the query
            ResultSet resultSet = statement.executeQuery(sql.toString());

            while (resultSet.next()) {
                entities entry = new entities();
                entry.name_e = resultSet.getString("obj_name");
                entry.date_e = resultSet.getString("date");
                entry.amount_e = Integer.parseInt(resultSet.getString("amount"));
                data.add(entry);
            }

            // Now the array should have 3 columns (name, date, and amount)
            String[][] array_data = new String[data.size()][3];

            for (int i = 0; i < data.size(); i++) { //convert the array list into an array
                entities entry = data.get(i);
                array_data[i][0] = entry.name_e;                   // Column 1: Name
                array_data[i][1] = entry.date_e;                   // Column 2: Date
                array_data[i][2] = String.valueOf(entry.amount_e); // Column 3: Amount
            }

            return array_data;

        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public static void exportToCSV(String filePath) {
        //method to export db to csv file
        Connection connection = null;
        Statement statement = null;
        FileWriter csvWriter = null;

        try {
            // 1. Establish database connection
            Class.forName("org.sqlite.JDBC");
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");
            statement = connection.createStatement();

            // 2. Execute query to retrieve data
            String query = "SELECT d_object.obj_name, obj_amt.date, obj_amt.amount " +
                    "FROM d_object, link_obj, obj_amt " +
                    "WHERE d_object.obj_id = link_obj.obj_id " +
                    "AND link_obj.date_id = obj_amt.date_id";

            ResultSet resultSet = statement.executeQuery(query);

            // 3. Create CSV Writer
            csvWriter = new FileWriter(filePath);

            // 4. Write CSV Header (assuming 3 columns: Name, Date, Amount)
            csvWriter.append("Name,Date,Amount\n");

            // 5. Write Data from ResultSet to CSV
            while (resultSet.next()) {
                String name = resultSet.getString("obj_name");
                String date = resultSet.getString("date");
                String amount = resultSet.getString("amount");

                // Escape the data for CSV formatting
                csvWriter.append(escapeCSV(name)).append(",")
                        .append(escapeCSV(date)).append(",")
                        .append(escapeCSV(amount)).append("\n");
            }

            System.out.println("CSV file created successfully at: " + filePath);

        } catch (Exception e) {
            throw new RuntimeException(e); //Exception handling

        } finally {
            try {
                // Close resources
                if (csvWriter != null) csvWriter.flush();
                if (csvWriter != null) csvWriter.close();
                if (statement != null) statement.close();
                if (connection != null) connection.close();
            } catch (IOException | SQLException ex) {
                System.out.println("Error: " + ex);
            }
        }
    }

    private static String escapeCSV(String data) {
        // Escape special characters for CSV (e.g., commas, quotes, newlines)
        if (data == null) {
            return "";  // Handle null values
        }
        String escapedData = data;

        // If data contains commas, quotes, or newlines, enclose it in double quotes
        if (data.contains(",") || data.contains("\"") || data.contains("\n")) {
            escapedData = "\"" + data.replace("\"", "\"\"") + "\"";
        }
        return escapedData;
    }
}