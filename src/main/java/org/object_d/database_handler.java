package org.object_d;

import org.tensorAction.detector;

import java.io.File;
import java.sql.*;
import java.util.ArrayList;

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
                String sql = "CREATE TABLE d_object (" +
                        "obj_id INTEGER NOT NULL, " +
                        "obj_name TEXT NOT NULL, " +
                        "PRIMARY KEY (obj_id))";

                // Create the `link_obj` table with a composite primary key and foreign key constraints
                String sql1 = "CREATE TABLE link_obj (" +
                        "obj_id INTEGER NOT NULL, " +
                        "date_id INTEGER NOT NULL, " +
                        "date DATE NOT NULL, " +
                        "PRIMARY KEY (obj_id, date_id), " +
                        "FOREIGN KEY (obj_id) REFERENCES d_object (obj_id))";

                // Create the `obj_amt` table with a foreign key constraint to `link_obj`
                String sql2 = "CREATE TABLE obj_amt (" +
                        "date_id INTEGER NOT NULL, " +
                        "amount INTEGER NOT NULL, " +
                        "PRIMARY KEY (date_id), " +
                        "FOREIGN KEY (date_id) REFERENCES link_obj (date_id))";

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
                String sql = "CREATE TABLE d_object (" +
                        "obj_id INTEGER NOT NULL, " +
                        "obj_name TEXT NOT NULL, " +
                        "PRIMARY KEY (obj_id))";

                // Create the `link_obj` table with a composite primary key and foreign key constraints
                String sql1 = "CREATE TABLE link_obj (" +
                        "obj_id INTEGER NOT NULL, " +
                        "date_id INTEGER NOT NULL, " +
                        "date DATE NOT NULL, " +
                        "PRIMARY KEY (obj_id, date_id), " +
                        "FOREIGN KEY (obj_id) REFERENCES d_object (obj_id))";

                // Create the `obj_amt` table with a foreign key constraint to `link_obj`
                String sql2 = "CREATE TABLE obj_amt (" +
                        "date_id INTEGER NOT NULL, " +
                        "amount INTEGER NOT NULL, " +
                        "PRIMARY KEY (date_id), " +
                        "FOREIGN KEY (date_id) REFERENCES link_obj (date_id))";

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

            String sql = "SELECT d_object.obj_name, link_obj.date, obj_amt.amount " +
                    "FROM d_object, link_obj, obj_amt " +
                    "WHERE d_object.obj_id = link_obj.obj_id " +
                    "AND link_obj.date_id = obj_amt.date_id " +
                    "ORDER BY d_object.obj_name ASC;";

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

    public static void addData(ArrayList<detector.entry> data) {   // {"item", "amount"}
        Connection connection = null;
        PreparedStatement preparedStatement = null;

        try {
            // Establish connection to the database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            //write statement for adding data

            System.out.println("Data added to the database");

        } catch (SQLException e) {
            throw new RuntimeException(e);

        } finally {
            // Close resources
            try {
                if (preparedStatement != null)
                    preparedStatement.close();
                if (connection != null)
                    connection.close();

            } catch (SQLException e) {
                System.out.println("Error occurred: " + e); //print out exception
            }
        }
    }
}