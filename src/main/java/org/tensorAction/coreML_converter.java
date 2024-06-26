package org.tensorAction;

import java.io.File;
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
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });

            // Move all files to the top-level directory and delete empty directories
            Files.walk(targetDir)
                    .filter(Files::isRegularFile)
                    .forEach(file -> {
                        try {
                            Path destination = targetDir.resolve(file.getFileName());
                            Files.move(file, destination, StandardCopyOption.REPLACE_EXISTING);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });

            // Delete empty directories
            Files.walk(targetDir)
                    .sorted(Comparator.reverseOrder())
                    .filter(Files::isDirectory)
                    .forEach(dir -> {
                        try {
                            if (!Files.list(dir).findAny().isPresent()) {
                                Files.delete(dir);
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });

            // Delete non-image files
            Files.walk(targetDir)
                    .filter(Files::isRegularFile)
                    .filter(file -> !isImageFile(file.toString()))
                    .forEach(file -> {
                        try {
                            Files.delete(file);
                        } catch (IOException e) {
                            e.printStackTrace();
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
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
        } catch (IOException e) {
            e.printStackTrace();
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
        //2.jpg [scissors,88.373215,205.99133,58.800613,173.9099][scissors,172.04897,208.84727,161.88799,275.86224]
        //1.jpg [scissors,173.99504,263.0,49.67339,295.79303]
        //image file "[" + detectedLabel + "," + yMin + "," + yMax + "," + xMin + "," + xMax + "]";
        for (int i = 0; i < labels.length; i++) {
            int label_amount;
            String filename;

            try {
                filename = labels[i].substring(0, labels[i].indexOf(" ")).trim();
                label_amount = (int) labels[i].chars().filter(ch -> ch == '[').count();
                String data = labels[i].substring(labels[i].indexOf("["));
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

                for (int j = 0; j < label_amount; j++) {
                    String[] parts = obj[j].split(",");

                    String label = parts[0].substring(1);
                    int yMin = (int) Double.parseDouble(parts[1]);
                    int yMax = (int) Double.parseDouble(parts[2]);
                    int xMin = (int) Double.parseDouble(parts[3]);
                    int xMax = (int) Double.parseDouble(parts[4].substring(0, parts[4].length() - 1));

                    int height = yMax - yMin;
                    int width = xMax - xMin;

                    System.out.println();
                    System.out.println("x " + (j + 1) + ": " + xMin);
                    System.out.println("y " + (j + 1) + ": " + yMax);
                    System.out.println("height " + (j + 1) + ": " + height);
                    System.out.println("width " + (j + 1) + ": " + width);
                    System.out.println("label " + (j + 1) + ": " + label);
                }

            } catch (StringIndexOutOfBoundsException e) {
                continue;
            }

            System.out.println("label amount " + label_amount);
            System.out.println("File name " + filename);
        }
    }

    public static void main(String[] args) {
        String[] data = {"2.jpg [idk,88.373215,205.99133,58.800613,173.9099][scissors,88.373215,205.99133,58.800613,173.9099][scissors,172.04897,208.84727,161.88799,275.86224]", "1.jpg [scissors,173.99504,263.0,49.67339,295.79303]", "3.jpg"};
        prepareOutput(data);
    }
}

//Create the json file and prepare output!!!