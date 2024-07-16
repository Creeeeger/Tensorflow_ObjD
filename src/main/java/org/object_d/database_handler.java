package org.object_d;

import java.io.File;
import java.sql.*;
import java.util.ArrayList;

public class database_handler {
    public static void CreateDatabase() {
        Connection connection = null;
        Statement statement = null;

        try {
            // Register JDBC driver
            Class.forName("org.sqlite.JDBC");

            // Check if the database file exists
            File dbFile = new File("results.db");
            if (dbFile.exists()) {
                // Open the existing database
                connection = DriverManager.getConnection("jdbc:sqlite:results.db");
                System.out.println("Database opened successfully");
            } else {
                // Create a new database
                connection = DriverManager.getConnection("jdbc:sqlite:results.db");
                statement = connection.createStatement();
                String sql = "CREATE TABLE IF NOT EXISTS results " +
                        "(id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                        "object TEXT NOT NULL, " +
                        "amount TEXT NOT NULL)";
                statement.executeUpdate(sql);
                System.out.println("Database created successfully");
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Close resources
            try {
                if (statement != null)
                    statement.close();
                if (connection != null)
                    connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }

        }
    }

    public static void resetDatabase() {
        Connection connection = null;
        Statement statement = null;

        try {
            // Establish connection to the database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Create a statement
            statement = connection.createStatement();

            // Drop the results table if it exists
            String dropTableSQL = "DROP TABLE IF EXISTS results";
            statement.executeUpdate(dropTableSQL);

            // Create the results table
            String createTableSQL = "CREATE TABLE results " +
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "object TEXT NOT NULL, " +
                    "amount TEXT NOT NULL)";
            statement.executeUpdate(createTableSQL);

            System.out.println("Database reset successful.");

        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources
            try {
                if (statement != null)
                    statement.close();
                if (connection != null)
                    connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    public static void addData(String[][] Data) {   // {"item", "amount"}
        Connection connection = null;
        PreparedStatement preparedStatement = null;

        try {
            // Establish connection to the database
            connection = DriverManager.getConnection("jdbc:sqlite:results.db");

            // Append each row of sample data to the end of the table
            for (String[] data : Data) {
                // Check if the object already exists in the database
                String selectSQL = "SELECT id, amount FROM results WHERE object = ?";
                preparedStatement = connection.prepareStatement(selectSQL);
                preparedStatement.setString(1, data[0]); // object
                ResultSet resultSet = preparedStatement.executeQuery();

                if (resultSet.next()) {
                    // If the object exists, update its amount by adding the new amount
                    int id = resultSet.getInt("id");
                    int currentAmount = Integer.parseInt(resultSet.getString("amount"));
                    int newAmount = Integer.parseInt(data[1]);
                    int updatedAmount = currentAmount + newAmount;

                    String updateSQL = "UPDATE results SET amount = ? WHERE id = ?";
                    preparedStatement = connection.prepareStatement(updateSQL);
                    preparedStatement.setString(1, String.valueOf(updatedAmount)); // updated amount
                    preparedStatement.setInt(2, id); // id of the existing entry
                    preparedStatement.executeUpdate();
                } else {
                    // If the object does not exist, insert it as a new entry
                    String insertSQL = "INSERT INTO results (object, amount) VALUES (?, ?)";
                    preparedStatement = connection.prepareStatement(insertSQL);
                    preparedStatement.setString(1, data[0]); // object
                    preparedStatement.setString(2, data[1]); // amount
                    preparedStatement.executeUpdate();
                }
            }

            System.out.println("Data appended to the end of the database.");

        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // Close resources
            try {
                if (preparedStatement != null)
                    preparedStatement.close();
                if (connection != null)
                    connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    public static String[][] readDatabase() {
        class entries {
            String name;
            int amount;
        }
        Connection connection;
        Statement statement;
        ArrayList<entries> data = new ArrayList<>();
        try {
            File dbfile = new File("results.db");
            if (dbfile.exists()) {
                connection = DriverManager.getConnection("jdbc:sqlite:results.db");
                statement = connection.createStatement();
                ResultSet resultSet = statement.executeQuery("SELECT object, amount FROM results");

                while (resultSet.next()) {
                    entries entry = new entries();
                    entry.name = resultSet.getString("object");
                    entry.amount = Integer.parseInt(resultSet.getString("amount"));
                    data.add(entry);
                }
                String[][] array_data = new String[data.size()][2];
                for (int i = 0; i < data.size(); i++) {
                    entries entry = data.get(i);
                    array_data[i][0] = entry.name;
                    array_data[i][1] = String.valueOf(entry.amount);
                }
                return array_data;
            } else {
                throw new Exception("File Not Found");
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return null;
    }
}