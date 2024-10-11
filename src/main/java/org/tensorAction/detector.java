package org.tensorAction;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.*;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.*;
import java.util.stream.Stream;

public class detector { // Class for detecting objects and labeling them as well as creating Data for the annotation process
    public final static String[] cocoLabels = new String[]{
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
            "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
            "hair brush"
    }; // Array of the labels used for identifying objects

    public static ArrayList<entry> classify(String imagePath, SavedModelBundle ModelBundle) {
        // Base logic for returning image path and associated labels
        ArrayList<entry> data = new ArrayList<>();

        // Create output directory "output_images" if it doesn't already exist
        File output_dir = new File("output_images");
        if (!output_dir.exists()) {
            output_dir.mkdir(); // Create the directory for storing annotated images
        }

        // Load OpenCV library to allow drawing of bounding boxes on the images
        nu.pattern.OpenCV.loadLocally();

        try (ModelBundle) {
            // Initialize the TensorFlow model bundle
            // Load COCO labels into a TreeMap for easy lookup during classification
            TreeMap<Float, String> cocoLabelMap = new TreeMap<>();
            float Label_count = 0;

            // Map each COCO label to a unique float key
            for (String cocoLabel : cocoLabels) {
                cocoLabelMap.put(Label_count, cocoLabel); // Store label with incremental float key
                Label_count++;
            }

            // Set up the TensorFlow graph and session for image processing
            try (Graph graph = new Graph()) {
                try (Session session = new Session(graph)) {
                    // Create an Ops object to represent TensorFlow operations
                    Ops operation = Ops.create(graph);

                    // Read and decode the JPEG image from the file system
                    DecodeJpeg decodeJpeg = operation.image.decodeJpeg(
                            operation.io.readFile(operation.constant(imagePath)).contents(), // Load image from the provided path
                            DecodeJpeg.channels(3L)); // Specify 3 color channels (RGB)

                    // Get the shape of the image (height, width, channels) from the decoded data
                    Shape imageShape = session.runner().fetch(decodeJpeg).run().get(0).shape();

                    // Reshape the image into a format compatible with the model's input (1 | height | width | channels)
                    Reshape<TUint8> reshape = operation.reshape(
                            decodeJpeg, // Tensor containing the decoded image data
                            operation.array(1, // First dimension is batch size (set to 1 for a single image)
                                    imageShape.asArray()[0], // Height of the image
                                    imageShape.asArray()[1], // Width of the image
                                    imageShape.asArray()[2]) // Number of color channels
                    );

                    // Get the reshaped tensor from the session
                    try (TUint8 reshape_Tensor = (TUint8) session.runner().fetch(reshape).run().get(0)) {
                        System.out.println(reshape_Tensor.shape()); // Debug: print shape

                        // Create a map to hold input tensors for the model
                        Map<String, Tensor> tensorMap = new HashMap<>();
                        tensorMap.put("input_tensor", reshape_Tensor);

                        // Get the result from the model using the function "serving_default"
                        try (Result result = ModelBundle.function("serving_default").call(tensorMap)) {
                            // Check if the model returned the expected results
                            if (result.get("detection_scores").isPresent() &&
                                    result.get("num_detections").isPresent() &&
                                    result.get("detection_classes").isPresent() &&
                                    result.get("detection_boxes").isPresent()) {

                                // Extract model results: scores, number of detections, classes, and bounding boxes
                                try (TFloat32 scores = (TFloat32) result.get("detection_scores").get();
                                     TFloat32 amount = (TFloat32) result.get("num_detections").get();
                                     TFloat32 classes = (TFloat32) result.get("detection_classes").get();
                                     TFloat32 boxes = (TFloat32) result.get("detection_boxes").get()) {

                                    // Number of detections (as int)
                                    int detections = (int) amount.getFloat(0);

                                    // ArrayList to hold the bounding boxes for visualization
                                    ArrayList<FloatNdArray> boxList = new ArrayList<>();

                                    // Proceed only if there are detections
                                    if (detections > 0) {
                                        // Get image dimensions (height and width)
                                        int imageHeight = (int) reshape_Tensor.shape().get(1);
                                        int imageWidth = (int) reshape_Tensor.shape().get(2);

                                        Mat image = Imgcodecs.imread(imagePath); // Read the image using OpenCV

                                        // Loop through all detected objects
                                        for (int i = 0; i < detections; i++) {
                                            // Get the confidence score for the current detection
                                            float score = scores.getFloat(0, i); // Detection confidence score

                                            // Only process objects with a confidence score higher than 30%
                                            if (score > 0.3f) {
                                                entry entry = new entry(); // Create a new entry for the detection

                                                // Get the bounding box coordinates for the detected object
                                                FloatNdArray boxFloat = boxes.get(0, i); // Bounding box coordinates
                                                boxList.add(boxFloat); // Store the bounding box

                                                // Calculate the bounding box coordinates scaled to the image dimensions
                                                float yMin = boxFloat.getFloat(0) * imageHeight; // Top boundary
                                                float xMin = boxFloat.getFloat(1) * imageWidth;  // Left boundary
                                                float yMax = boxFloat.getFloat(2) * imageHeight; // Bottom boundary
                                                float xMax = boxFloat.getFloat(3) * imageWidth;  // Right boundary

                                                // Log the bounding box coordinates for debugging purposes
                                                System.out.println("Log: box coordinates: [yMin: " + yMin + ", xMin: " + xMin + ", yMax: " + yMax + ", xMax: " + xMax + "]");

                                                // Get the class index of the detected object and map it to a label
                                                float classIndex = classes.getFloat(0, i); // Detected class index
                                                int number_class = ((int) classIndex) - 1; // Adjust index for COCO label mapping

                                                // Ensure the class index doesn't go out of bounds if we hit the last entry then we shift it to the first one
                                                if (number_class < 0) {
                                                    number_class = 0;
                                                }

                                                // Get the label for the detected class
                                                String detectedLabel = cocoLabels[number_class]; // Label for the detected object class
                                                System.out.println("Detected: " + detectedLabel + " with score: " + String.format("%.2f", (score * 100)) + "%.");

                                                // Draw the bounding box on the image using OpenCV
                                                Imgproc.rectangle(image, new Point(xMin, yMin), new Point(xMax, yMax), new Scalar(0, 255, 0), 1); // Green box

                                                // Add the detected label and confidence score on the image
                                                Imgproc.putText(image, detectedLabel + String.format(" %.2f", (score * 100)) + "%", new Point(xMin, yMin - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 0, 0), 1); // Text label

                                                // Fill the entry with the detected label, current timestamp, and confidence score
                                                entry.label = detectedLabel; // Set label
                                                entry.date = new java.sql.Date(Date.from(Instant.now()).getTime()); // Set current date
                                                entry.percentage = Float.parseFloat(String.format("%.2f", score * 100)); // Set confidence score as percentage

                                                // Add the entry to the data list for later use
                                                data.add(entry);
                                            }
                                        }

                                        // Save the annotated image with bounding boxes
                                        String outputImagePath = "output_images/annotated_" + new File(imagePath).getName();
                                        Imgcodecs.imwrite(outputImagePath, image); // Save the image

                                        // Add the image path to a new entry
                                        entry entry = new entry();
                                        entry.imagePath = outputImagePath;
                                        data.add(entry); // Add to data list

                                    } else {
                                        // If no detections, return the original image path for just displaying an unlabeled image
                                        entry entry = new entry();
                                        entry.imagePath = imagePath;
                                        data.add(entry);
                                        return data;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e); // Handle exceptions
        }
        return data; // Return the list of entries
    }

    public String[] label(String imagePath, String TensorPath) {
        try (Stream<Path> paths = Files.walk(Paths.get(imagePath))) {
            // get all the files and paths
            // Filter out files that are not image files and only keep those with .jpg, .png, or .jpeg extensions
            List<Path> imageFiles = paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".jpg") || p.toString().endsWith(".png") || p.toString().endsWith(".jpeg")) // Can add more formats on demand
                    .toList();

            // Initialize an array to store the results (one entry per image)
            String[] returnArray = new String[imageFiles.size()];

            // Loop through each image file found in the directory
            for (int i = 0; i < imageFiles.size(); i++) {
                // Load the TensorFlow SavedModel for inference from the path
                SavedModelBundle ModelBundle = SavedModelBundle.load(TensorPath, "serve");

                // Get the string representation of the current image file
                String imageFile = imageFiles.get(i).toString();
                StringBuilder returnString = new StringBuilder(); // To store the detection results for the image

                try (ModelBundle) {
                    // Initialize the model and prepare the COCO label map
                    TreeMap<Float, String> cocoLabelMap = new TreeMap<>();
                    float Label_count = 0;

                    // Fill the label map with COCO class names, mapping float keys to label names
                    for (String cocoLabel : cocoLabels) {
                        cocoLabelMap.put(Label_count, cocoLabel);
                        Label_count++;
                    }

                    // Set up TensorFlow graph for image processing
                    try (Graph graph = new Graph()) {
                        // Create a TensorFlow session within the graph to run operations
                        try (Session session = new Session(graph)) {
                            Ops operation = Ops.create(graph);

                            // Decode the image file into a TensorFlow tensor with RGB channels (3 channels)
                            DecodeJpeg decodeJpeg = operation.image.decodeJpeg(
                                    operation.io.readFile(operation.constant(imageFile)).contents(),
                                    DecodeJpeg.channels(3L));

                            // Fetch the image shape (height, width, and color channels)
                            Shape imageShape = session.runner().fetch(decodeJpeg).run().get(0).shape();

                            // Reshape the image tensor for further processing, so it's in the format: [1, height, width, channels]
                            Reshape<TUint8> reshape = operation.reshape(decodeJpeg,
                                    operation.array(1, imageShape.asArray()[0], imageShape.asArray()[1], imageShape.asArray()[2]));
                            // The reshaped tensor will have 1 image, with its height, width, and 3 color channels

                            // Reshape operations now, we need to cast because we need integer format from the tensor
                            try (TUint8 reshape_Tensor = (TUint8) session.runner().fetch(reshape).run().get(0)) {

                                // Create a HashMap to store the reshaped image tensor as the input for the model
                                Map<String, Tensor> tensorMap = new HashMap<>();
                                tensorMap.put("input_tensor", reshape_Tensor);

                                // Run the model using the "serving_default" function, which processes the image and returns detections
                                try (Result result = ModelBundle.function("serving_default").call(tensorMap)) {

                                    // Ensure the model returned the expected output tensors for scores, detections, classes, and boxes
                                    if (result.get("detection_scores").isPresent() &&
                                            result.get("num_detections").isPresent() &&
                                            result.get("detection_classes").isPresent() &&
                                            result.get("detection_boxes").isPresent()) {

                                        // Extract the model output tensors for scores, detection count, classes, and bounding boxes
                                        try (TFloat32 scores = (TFloat32) result.get("detection_scores").get();
                                             TFloat32 amount = (TFloat32) result.get("num_detections").get();
                                             TFloat32 classes = (TFloat32) result.get("detection_classes").get();
                                             TFloat32 boxes = (TFloat32) result.get("detection_boxes").get()) {

                                            // Get the total number of detected objects from the model's output
                                            int detections = (int) amount.getFloat(0);

                                            // Create a list to store bounding box coordinates
                                            ArrayList<FloatNdArray> boxList = new ArrayList<>();
                                            returnString = new StringBuilder(); // Reset the return string for the current image

                                            // Proceed only if the model detected at least one object
                                            if (detections > 0) {
                                                // Get the dimensions of the image (height and width) from the reshaped tensor
                                                int imageHeight = (int) reshape_Tensor.shape().get(1); // Extract the image height
                                                int imageWidth = (int) reshape_Tensor.shape().get(2);  // Extract the image width

                                                // Loop through all detected objects
                                                for (int j = 0; j < detections; j++) {
                                                    // Get the confidence score of each detected object
                                                    float score = scores.getFloat(0, j);

                                                    // Filter out objects with less than 30% detection probability
                                                    if (score > 0.3f) {
                                                        // Get the bounding box coordinates for each detected object
                                                        FloatNdArray boxFloat = boxes.get(0, j);
                                                        boxList.add(boxFloat); // Add the bounding box coordinates to the list for coordinate extraction

                                                        // Calculate the bounding box coordinates relative to the image size
                                                        float yMin = boxFloat.getFloat(0) * imageHeight; // Top coordinate
                                                        float xMin = boxFloat.getFloat(1) * imageWidth;  // Left coordinate
                                                        float yMax = boxFloat.getFloat(2) * imageHeight; // Bottom coordinate
                                                        float xMax = boxFloat.getFloat(3) * imageWidth;  // Right coordinate

                                                        // Get the detected class index and map it to the corresponding label
                                                        float classIndex = classes.getFloat(0, j);
                                                        int number_class = ((int) classIndex) - 1; // Adjust the class index

                                                        // Ensure the class index doesn't go out of bounds if we hit the last entry then we shift it to the first one
                                                        if (number_class < 0) {
                                                            number_class = 0;
                                                        }

                                                        // Retrieve the label for the detected class from cocoLabels
                                                        String detectedLabel = cocoLabels[number_class];

                                                        // Append the detected label and bounding box coordinates to the return string for data extraction later
                                                        returnString.append("[")
                                                                .append(detectedLabel)
                                                                .append(",").append(yMin)
                                                                .append(",").append(yMax)
                                                                .append(",").append(xMin)
                                                                .append(",").append(xMax)
                                                                .append("]");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Assign the formatted detection results to the returnArray for the current image
                returnArray[i] = imageFile.substring(imageFile.indexOf("/") + 1) + " " + returnString;

                // Print the result for the current image
                System.out.println(returnArray[i]);
            }
            // Return the array containing detection results for all images
            return returnArray;

        } catch (IOException e) {
            // Handle any IO exceptions by throwing a RuntimeException
            throw new RuntimeException(e);
        }
    }

    // Create a class to store image data entries
    public static class entry {

        // Variables to hold the image path, label, date, and accuracy
        String imagePath;    // Path to the image file
        String label;        // Label for the detected object in the image
        java.sql.Date date;  // Date when the image was detected
        float percentage;    // Percentage related to the accuracy

        // Getter for the image path
        public String getImagePath() {
            return imagePath; // Return the image file path
        }

        // Getter for the label
        public String getLabel() {
            return label; // Return the label associated with the image
        }

        // Setter for the label
        public void setLabel(String label) {
            this.label = label; // Set the label
        }

        // Getter for the date
        public java.sql.Date getDate() {
            return date; // Return the date when the entry was added
        }

        // Setter for the date
        public void setDate(java.sql.Date date) {
            this.date = date; // Set the date for the entry
        }

        // Getter for the percentage
        public float getPercentage() {
            return percentage; // Return the percentage value for the accuracy of the label
        }
    }
}