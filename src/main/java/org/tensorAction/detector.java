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
import java.util.*;
import java.util.stream.Stream;

public class detector {
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
    };

    public static String[] classify(String imagePath, SavedModelBundle ModelBundle) {
        //Base logic for returning image path and labels
        String[] returnArray = new String[2];
        returnArray[0] = imagePath;
        returnArray[1] = "";

        //Create output directory
        File output_dir = new File("output_images");
        if (!output_dir.exists()) {
            output_dir.mkdir();
        }
        nu.pattern.OpenCV.loadLocally();

        try (ModelBundle) {
            //initialise model bundle
            //Load our labels in
            TreeMap<Float, String> cocoLabelMap = new TreeMap<>();
            float Label_count = 0;

            for (String cocoLabel : cocoLabels) {
                cocoLabelMap.put(Label_count, cocoLabel);
                Label_count++;
            }

            //Setup graph and session and operation graph
            try (Graph graph = new Graph()) {
                try (Session session = new Session(graph)) {
                    Ops operation = Ops.create(graph);

                    //Decode the jpeg and get the file
                    DecodeJpeg decodeJpeg = operation.image.decodeJpeg(operation.io.readFile(operation.constant(imagePath)).contents(), DecodeJpeg.channels(3L));

                    //Get the shape of the image
                    Shape imageShape = session.runner().fetch(decodeJpeg).run().get(0).shape();

                    //Now we got to reshape it as we saw over debugging that its in an unusable shape
                    Reshape<TUint8> reshape = operation.reshape(decodeJpeg, operation.array(1,
                            imageShape.asArray()[0],
                            imageShape.asArray()[1],
                            imageShape.asArray()[2]
                    )); //shape is in form of 1|height|width|color channels

                    //Reshape operations now, we need to cast because we need integer format from the tensor
                    try (TUint8 reshape_Tensor = (TUint8) session.runner().fetch(reshape).run().get(0)) {
                        System.out.println(reshape_Tensor.shape()); //check the shape

                        //Create hashmap and add our image as input tensor to it
                        Map<String, Tensor> tensorMap = new HashMap<>();
                        tensorMap.put("input_tensor", reshape_Tensor);

                        //Set up the result operations
                        try (Result result = ModelBundle.function("serving_default").call(tensorMap)) {
                            if (result.get("detection_scores").isPresent() &&
                                    result.get("num_detections").isPresent() &&
                                    result.get("detection_classes").isPresent() &&
                                    result.get("detection_boxes").isPresent()) {

                                //Set up the functions of the model we use
                                try (TFloat32 scores = (TFloat32) result.get("detection_scores").get();
                                     TFloat32 amount = (TFloat32) result.get("num_detections").get();
                                     TFloat32 classes = (TFloat32) result.get("detection_classes").get();
                                     TFloat32 boxes = (TFloat32) result.get("detection_boxes").get()) {

                                    //Get the amount of detections we got and cast it
                                    int detections = (int) amount.getFloat(0);

                                    //Array for boxes for visualising objects later with open cv
                                    ArrayList<FloatNdArray> boxList = new ArrayList<>();

                                    //only proceed when we got more than 0 detections
                                    if (detections > 0) {
                                        //Get the image dimensions
                                        int imageHeight = (int) reshape_Tensor.shape().get(1);
                                        int imageWidth = (int) reshape_Tensor.shape().get(2);

                                        Mat image = Imgcodecs.imread(imagePath);

                                        //Loop through all detected objects
                                        for (int i = 0; i < detections; i++) {
                                            //get the score of each object
                                            float score = scores.getFloat(0, i);

                                            //Just take objects with 30% or higher chance
                                            if (score > 0.3f) {
                                                //get the boxes where the objects are
                                                FloatNdArray boxFloat = boxes.get(0, i);
                                                boxList.add(boxFloat);

                                                // Print the coordinates of the box
                                                float yMin = boxFloat.getFloat(0) * imageHeight;
                                                float xMin = boxFloat.getFloat(1) * imageWidth;
                                                float yMax = boxFloat.getFloat(2) * imageHeight;
                                                float xMax = boxFloat.getFloat(3) * imageWidth;
                                                System.out.println("Box coordinates: [yMin: " + yMin + ", xMin: " + xMin + ", yMax: " + yMax + ", xMax: " + xMax + "]");

                                                // Get the detected class index and map it to the corresponding label
                                                float classIndex = classes.getFloat(0, i);
                                                int number_class = ((int) classIndex) - 1;

                                                if (number_class < 0) {
                                                    number_class = 0;
                                                }

                                                String detectedLabel = cocoLabels[number_class];
                                                System.out.println("Detected: " + detectedLabel + " with score: " + String.format("%.2f", (score * 100)) + "%.");
                                                returnArray[1] = returnArray[1] + detectedLabel + ": " + String.format("%.2f", (score * 100)) + "%\n";

                                                // Draw the rectangle on the image
                                                Imgproc.rectangle(image, new Point(xMin, yMin), new Point(xMax, yMax), new Scalar(0, 0, 0), 1);

                                                // Optionally, you can put the label text on the image
                                                Imgproc.putText(image, detectedLabel + String.format(" %.2f", (score * 100)) + "%", new Point(xMin, yMin - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 0, 0), 1);
                                            }
                                        }

                                        // Save the modified image to the output directory
                                        String outputImagePath = "output_images/annotated_" + new File(imagePath).getName();
                                        Imgcodecs.imwrite(outputImagePath, image);

                                        // Update the return array to include the path to the annotated image
                                        returnArray[0] = outputImagePath;

                                    } else {
                                        returnArray[0] = imagePath;
                                        returnArray[1] = "Nothing detected";
                                        return returnArray;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return returnArray;
        }
    }

    public String[] label(String imagePath, String TensorPath) {
        try (Stream<Path> paths = Files.walk(Paths.get(imagePath))) {
            // Filter only regular image files (you might need to adjust this based on your file types)
            List<Path> imageFiles = paths.filter(Files::isRegularFile).filter(p -> p.toString().endsWith(".jpg") || p.toString().endsWith(".png") || p.toString().endsWith(".jpeg")).toList();

            String[] returnArray = new String[imageFiles.size()];

            for (int i = 0; i < imageFiles.size(); i++) {
                SavedModelBundle ModelBundle = SavedModelBundle.load(TensorPath, "serve");

                String imageFile = imageFiles.get(i).toString();
                StringBuilder returnString = new StringBuilder();

                try (ModelBundle) {
                    //initialise model bundle
                    //Load our labels in
                    TreeMap<Float, String> cocoLabelMap = new TreeMap<>();
                    float Label_count = 0;

                    for (String cocoLabel : cocoLabels) {
                        cocoLabelMap.put(Label_count, cocoLabel);
                        Label_count++;
                    }

                    //Setup graph and session and operation graph
                    try (Graph graph = new Graph()) {
                        try (Session session = new Session(graph)) {
                            Ops operation = Ops.create(graph);

                            //Decode the jpeg and get the file
                            DecodeJpeg decodeJpeg = operation.image.decodeJpeg(operation.io.readFile(operation.constant(imageFile)).contents(), DecodeJpeg.channels(3L));

                            //Get the shape of the image
                            Shape imageShape = session.runner().fetch(decodeJpeg).run().get(0).shape();

                            //Now we got to reshape it as we saw over debugging that its in an unusable shape
                            Reshape<TUint8> reshape = operation.reshape(decodeJpeg, operation.array(1,
                                    imageShape.asArray()[0],
                                    imageShape.asArray()[1],
                                    imageShape.asArray()[2]
                            )); //shape is in form of 1|height|width|color channels

                            //Reshape operations now, we need to cast because we need integer format from the tensor
                            try (TUint8 reshape_Tensor = (TUint8) session.runner().fetch(reshape).run().get(0)) {

                                //Create hashmap and add our image as input tensor to it
                                Map<String, Tensor> tensorMap = new HashMap<>();
                                tensorMap.put("input_tensor", reshape_Tensor);

                                //Set up the result operations
                                try (Result result = ModelBundle.function("serving_default").call(tensorMap)) {

                                    if (result.get("detection_scores").isPresent() &&
                                            result.get("num_detections").isPresent() &&
                                            result.get("detection_classes").isPresent() &&
                                            result.get("detection_boxes").isPresent()) {

                                        //Set up the functions of the model we use
                                        try (TFloat32 scores = (TFloat32) result.get("detection_scores").get();
                                             TFloat32 amount = (TFloat32) result.get("num_detections").get();
                                             TFloat32 classes = (TFloat32) result.get("detection_classes").get();
                                             TFloat32 boxes = (TFloat32) result.get("detection_boxes").get()) {

                                            //Get the amount of detections we got and cast it
                                            int detections = (int) amount.getFloat(0);

                                            //Array for boxes for visualising objects later with open cv
                                            ArrayList<FloatNdArray> boxList = new ArrayList<>();
                                            returnString = new StringBuilder();

                                            //only proceed when we got more than 0 detections
                                            if (detections > 0) {
                                                //Get the image dimensions
                                                int imageHeight = (int) reshape_Tensor.shape().get(1);
                                                int imageWidth = (int) reshape_Tensor.shape().get(2);

                                                //Loop through all detected objects
                                                for (int j = 0; j < detections; j++) {
                                                    //get the score of each object
                                                    float score = scores.getFloat(0, j);

                                                    //Just take objects with 30% or higher chance
                                                    if (score > 0.3f) {
                                                        //get the boxes where the objects are
                                                        FloatNdArray boxFloat = boxes.get(0, j);
                                                        boxList.add(boxFloat);

                                                        // Print the coordinates of the box
                                                        float yMin = boxFloat.getFloat(0) * imageHeight;
                                                        float xMin = boxFloat.getFloat(1) * imageWidth;
                                                        float yMax = boxFloat.getFloat(2) * imageHeight;
                                                        float xMax = boxFloat.getFloat(3) * imageWidth;

                                                        // Get the detected class index and map it to the corresponding label
                                                        float classIndex = classes.getFloat(0, j);
                                                        int number_class = ((int) classIndex) - 1;

                                                        if (number_class < 0) {
                                                            number_class = 0;
                                                        }

                                                        String detectedLabel = cocoLabels[number_class];
                                                        returnString.append("[").append(detectedLabel).append(",").append(yMin).append(",").append(yMax).append(",").append(xMin).append(",").append(xMax).append("]");
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
                returnArray[i] = imageFile.substring(imageFile.indexOf("/") + 1) + " " + returnString;
                System.out.println(returnArray[i]);
            }
            return returnArray;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}