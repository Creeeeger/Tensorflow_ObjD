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
            // Create the target directory
            Path targetDir = Paths.get("CoreML_out");

            // Delete the target directory if it exists
            if (Files.exists(targetDir)) {
                deleteDirectory(targetDir);
            }
            if (!Files.exists(targetDir)) {
                Files.createDirectories(targetDir);
            }

            // Copy the folder from the source path to the target directory
            Path sourceFolder = Paths.get(sourceFolderPath);
            Files.walk(sourceFolder)
                    .sorted(Comparator.naturalOrder()) // Sort alphabetically
                    .forEach(source -> {
                        try {
                            Path destination = targetDir.resolve(sourceFolder.relativize(source));
                            if (Files.isDirectory(source)) {
                                if (!Files.exists(destination)) {
                                    Files.createDirectories(destination);
                                }
                            } else {
                                Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
                            }
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Move all files to the top-level directory and delete empty directories
            Files.walk(targetDir)
                    .filter(Files::isRegularFile)
                    .forEach(file -> {
                        try {
                            Path destination = targetDir.resolve(file.getFileName());
                            Files.move(file, destination, StandardCopyOption.REPLACE_EXISTING);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Delete empty directories
            Files.walk(targetDir)
                    .sorted(Comparator.reverseOrder())
                    .filter(Files::isDirectory)
                    .forEach(dir -> {
                        try {
                            if (Files.list(dir).findAny().isEmpty()) {
                                Files.delete(dir);
                            }
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Delete non-image files
            Files.walk(targetDir)
                    .filter(Files::isRegularFile)
                    .filter(file -> !isImageFile(file.toString()))
                    .forEach(file -> {
                        try {
                            Files.delete(file);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

            // Rename image files numerically
            AtomicInteger fileCounter = new AtomicInteger(1);
            Files.walk(targetDir)
                    .sorted(Comparator.naturalOrder()) // Sort alphabetically
                    .filter(Files::isRegularFile)
                    .forEach(file -> {
                        try {
                            String fileExtension = getFileExtension(file.toString());
                            String newFileName = fileCounter.getAndIncrement() + fileExtension;
                            Files.move(file, targetDir.resolve(newFileName), StandardCopyOption.REPLACE_EXISTING);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static String getFileExtension(String fileName) {
        int lastIndexOfDot = fileName.lastIndexOf('.');
        return (lastIndexOfDot == -1) ? "" : fileName.substring(lastIndexOfDot);
    }

    private static boolean isImageFile(String fileName) {
        String fileExtension = getFileExtension(fileName).toLowerCase();
        return fileExtension.equals(".jpg") || fileExtension.equals(".jpeg") || fileExtension.equals(".png");
    }

    private static void deleteDirectory(Path path) throws IOException {
        Files.walk(path)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
    }

    public static void labelImages(String modelBundle_path) {
        detector detector = new detector();
        String[] labels = detector.label("CoreML_out", modelBundle_path);
        prepareOutput(labels);
    }

    public static void prepareOutput(String[] labels) {
        JSONArray jsonArray = new JSONArray();

        for (String label : labels) {
            try {
                String filename = label.substring(0, label.indexOf(" ")).trim();
                int label_amount = (int) label.chars().filter(ch -> ch == '[').count();
                String data = label.substring(label.indexOf("["));
                String[] obj = new String[label_amount];

                int startIndex = 0;
                int endIndex = data.indexOf("]");
                for (int j = 0; j < label_amount; j++) {
                    obj[j] = data.substring(startIndex, endIndex + 1);
                    startIndex = endIndex + 1;
                    if (startIndex < data.length()) {
                        endIndex = data.indexOf("]", startIndex);
                        if (endIndex == -1) {
                            endIndex = data.length();
                        }
                    }
                }

                JSONArray annotations = new JSONArray();
                for (String s : obj) {
                    String[] parts = s.split(",");

                    String labelName = parts[0].substring(1);
                    int yMin = (int) Double.parseDouble(parts[1]);
                    int yMax = (int) Double.parseDouble(parts[2]);
                    int xMin = (int) Double.parseDouble(parts[3]);
                    int xMax = (int) Double.parseDouble(parts[4].substring(0, parts[4].length() - 1));

                    int height = yMax - yMin;
                    int width = xMax - xMin;
                    int xCenter = xMin + width / 2;
                    int yCenter = yMin + height / 2;

                    JSONObject annotation = new JSONObject();
                    JSONObject coordinates = new JSONObject();
                    coordinates.put("y", yCenter);
                    coordinates.put("x", xCenter);
                    coordinates.put("height", height);
                    coordinates.put("width", width);
                    annotation.put("coordinates", coordinates);
                    annotation.put("label", labelName);
                    annotations.put(annotation);
                }

                JSONObject jsonObject = new JSONObject();
                jsonObject.put("imagefilename", filename);
                jsonObject.put("annotation", annotations);
                jsonArray.put(jsonObject);

            } catch (StringIndexOutOfBoundsException e) {
                throw new RuntimeException(e);
            }
        }

        File file = new File("CoreML_out/Annotations.json");
        try {
            file.createNewFile();
            FileWriter fileWriter = new FileWriter(file);
            fileWriter.write(jsonArray.toString(4));
            fileWriter.close();
        } catch (Exception x) {
            System.out.println(x.getMessage());
        }
    }
}