package org.tensorAction;

import org.tensorflow.SavedModelBundle;

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
        return fileExtension.equals(".jpg") || fileExtension.equals(".jpeg") || fileExtension.equals(".png")
                || fileExtension.equals(".gif") || fileExtension.equals(".bmp") || fileExtension.equals(".tiff");
    }

    private static void deleteDirectory(Path path) throws IOException {
        Files.walk(path)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
    }

    public static void labelImages(SavedModelBundle modelBundle) {

        detector detector = new detector();
        String[] labels = detector.label("CoreML_out", modelBundle);
        prepareOutput(labels);
    }

    public static void prepareOutput(String[] labels) {

    }

    public static void main(String[] args) {
        organiseFiles("/Users/gregor/Desktop/Tensorflow_ObjD/flower_photos");
    }
}