package org.tensorAction;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;

public class coreML_converter {

    public static void organiseFiles(String sourceFolderPath) {
        try {
            // Create the target directory path for organized files
            Path targetDir = Paths.get("CoreML_out");

            // Delete the target directory if it exists to start fresh
            if (Files.exists(targetDir)) {
                deleteDirectory(targetDir);
            }

            // Create the target directory if it does not exist
            if (!Files.exists(targetDir)) {
                Files.createDirectories(targetDir);
            }

            // Copy files from the source folder to the target directory
            Path sourceFolder = Paths.get(sourceFolderPath);
            Files.walk(sourceFolder)
                    .sorted(Comparator.naturalOrder()) // Sort files alphabetically
                    .forEach(source -> {
                        try {
                            // Determine the destination path for each file
                            Path destination = targetDir.resolve(sourceFolder.relativize(source));
                            if (Files.isDirectory(source)) {
                                // Create the destination directory if it doesn't exist
                                if (!Files.exists(destination)) {
                                    Files.createDirectories(destination);
                                }
                            } else {
                                // Copy the file to the destination, replacing if it exists
                                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
                            }
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Move all files to the top-level directory and delete empty directories
            Files.walk(targetDir)
                    .filter(Files::isRegularFile) // Only regular files
                    .forEach(file -> {
                        try {
                            // Move each file to the top-level directory of the target
                            Path destination = targetDir.resolve(file.getFileName());
                            Files.move(file, destination, StandardCopyOption.REPLACE_EXISTING);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Delete any empty directories after moving files
            Files.walk(targetDir)
                    .sorted(Comparator.reverseOrder()) // Sort to delete leaf directories first
                    .filter(Files::isDirectory) // Only directories
                    .forEach(dir -> {
                        try {
                            // Check if the directory is empty and delete if it is
                            if (Files.list(dir).findAny().isEmpty()) {
                                Files.delete(dir);
                            }
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Delete all non-image files from the target directory
            Files.walk(targetDir)
                    .filter(Files::isRegularFile) // Only regular files
                    .filter(file -> !isImageFile(file.toString())) // Filter out non-image files
                    .forEach(file -> {
                        try {
                            // Delete the non-image file
                            Files.delete(file);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Rename remaining image files numerically
            AtomicInteger fileCounter = new AtomicInteger(1);
            Files.walk(targetDir)
                    .sorted(Comparator.naturalOrder()) // Sort files alphabetically
                    .filter(Files::isRegularFile) // Only regular files
                    .forEach(file -> {
                        try {
                            String fileExtension = getFileExtension(file.toString()); // Get the file extension
                            // Create a new file name based on the counter value
                            String newFileName = fileCounter.getAndIncrement() + fileExtension;
                            // Rename the file to the new file name
                            Files.move(file, targetDir.resolve(newFileName), StandardCopyOption.REPLACE_EXISTING);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });
        } catch (Exception e) {
            throw new RuntimeException(e); // Handle any exceptions by throwing a runtime exception with exception as message
        }
    }

    // Method to get the file extension from the given file name
    private static String getFileExtension(String fileName) {
        int lastIndexOfDot = fileName.lastIndexOf('.'); // Find the last index of the dot character
        return (lastIndexOfDot == -1) ? "" : fileName.substring(lastIndexOfDot); // Return the extension or empty string if no dot found
    }

    // Method to check if the provided file name corresponds to an image file
    private static boolean isImageFile(String fileName) {
        String fileExtension = getFileExtension(fileName).toLowerCase(); // Get the file extension in lowercase
        // Check if the extension matches supported image formats
        return fileExtension.equals(".jpg") || fileExtension.equals(".jpeg") || fileExtension.equals(".png");
    }

    // Method to delete a directory and all its contents recursively
    private static void deleteDirectory(Path path) throws IOException {
        // Walk the file tree, sorting paths in reverse order to ensure files are deleted before their directories
        Files.walk(path)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile) // Convert Paths to File objects
                .forEach(File::delete); // Delete each file and directory
    }

    // Method to label images using a specified model
    public static void labelImages(String modelBundle_path) {
        detector detector = new detector(); // Create an instance of the detector class
        // Call the label method of the detector, passing the output directory and model path
        String[] labels = detector.label("CoreML_out", modelBundle_path);
        prepareOutput(labels); // Prepare the output JSON based on the obtained labels
    }

    // Method to prepare and write the output annotations to a JSON file
    public static void prepareOutput(String[] labels) {
        JSONArray jsonArray = new JSONArray(); // Create a JSON array to hold all image annotations

        for (String label : labels) {
            try {
                // Extract the filename and the data related to the image
                String filename = label.substring(0, label.indexOf(" ")).trim(); // Get the image filename
                int label_amount = (int) label.chars().filter(ch -> ch == '[').count(); // Count how many annotations there are
                String data = label.substring(label.indexOf("[")); // Get the substring starting from the first '['
                String[] obj = new String[label_amount]; // Array to hold the parsed annotation objects

                int startIndex = 0;
                int endIndex = data.indexOf("]"); // Find the first closing bracket

                // Extract annotation data
                for (int j = 0; j < label_amount; j++) {
                    obj[j] = data.substring(startIndex, endIndex + 1); // Get the current annotation
                    startIndex = endIndex + 1; // Move start index to the next character
                    if (startIndex < data.length()) {
                        endIndex = data.indexOf("]", startIndex); // Find the next closing bracket
                        if (endIndex == -1) {
                            endIndex = data.length(); // If no closing bracket is found, go to end of data
                        }
                    }
                }

                JSONArray annotations = new JSONArray(); // Create a new JSON array for the current image's annotations
                for (String s : obj) { // Loop through each annotation object
                    String[] parts = s.split(","); // Split the string by commas

                    // Extract label and coordinates
                    String labelName = parts[0].substring(1); // Get the label name, removing the leading quote
                    int yMin = (int) Double.parseDouble(parts[1]);
                    int yMax = (int) Double.parseDouble(parts[2]);
                    int xMin = (int) Double.parseDouble(parts[3]);
                    int xMax = (int) Double.parseDouble(parts[4].substring(0, parts[4].length() - 1));

                    // Calculate height, width, and center coordinates
                    int height = yMax - yMin;
                    int width = xMax - xMin;
                    int xCenter = xMin + width / 2;
                    int yCenter = yMin + height / 2;

                    // Create JSON objects for the annotation
                    JSONObject annotation = new JSONObject();
                    JSONObject coordinates = new JSONObject();
                    coordinates.put("y", yCenter); // Y center coordinate
                    coordinates.put("x", xCenter); // X center coordinate
                    coordinates.put("height", height); // Height of the bounding box
                    coordinates.put("width", width); // Width of the bounding box
                    annotation.put("coordinates", coordinates); // Add coordinates to annotation
                    annotation.put("label", labelName); // Add label to annotation
                    annotations.put(annotation); // Add annotation to the annotations array
                }

                // Create a JSON object to represent the current image and its annotations
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("imagefilename", filename); // Add the image filename
                jsonObject.put("annotation", annotations); // Add annotations
                jsonArray.put(jsonObject); // Add the image object to the main JSON array

            } catch (StringIndexOutOfBoundsException e) {
                throw new RuntimeException(e); // Handle any string indexing errors
            }
        }

        // Write the JSON array to a file
        File file = new File("CoreML_out/Annotations.json"); // Define the output file
        System.out.println("Annotation process has finished"); //give an output of the process

        try {
            file.createNewFile(); // Create the file if it doesn't exist
            FileWriter fileWriter = new FileWriter(file); // Create a FileWriter for the output file
            fileWriter.write(jsonArray.toString(4)); // Write the JSON array with an indentation of 4 spaces
            fileWriter.close(); // Close the FileWriter
        } catch (Exception x) {
            System.out.println(x.getMessage()); // Handle any exceptions during file writing
        }
    }
}