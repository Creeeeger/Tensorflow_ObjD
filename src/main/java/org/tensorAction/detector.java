package org.tensorAction;

import org.tensorflow.*;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.io.IOException;
import java.nio.file.*;
import java.util.stream.Stream;

public class detector {
    private final static String[] cocoLabels = new String[]{
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

    public String[] classify(String imagePath, SavedModelBundle ModelBundle) {

        //Base logic for returning image path and labels
        String[] returnArray = new String[2];
        returnArray[0] = imagePath;
        returnArray[1] = "";

        //Create output directory
        File output_dir = new File("output_images");
        if (!output_dir.exists()) {
            output_dir.mkdir();
        }
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

                    //Setup the sessions and runners
                    Ops operation = Ops.create(graph);
                    Session.Runner runner = session.runner();

                    //get the file from the file path
                    Operand<TString> filePath = operation.constant(imagePath);
                    ReadFile file = operation.io.readFile(filePath);

                    //Decode the jpeg
                    DecodeJpeg.Options jpeg_options = DecodeJpeg.channels(3L);
                    DecodeJpeg decodeJpeg = operation.image.decodeJpeg(file.contents(), jpeg_options);

                    //Get the shape of the image
                    Shape imageShape = runner.fetch(decodeJpeg).run().get(0).shape();

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

                        //Setup the result operations
                        Result result = ModelBundle.function("serving_default").call(tensorMap);
                        if (result.get("detection_scores").isPresent() &&
                                result.get("num_detections").isPresent() &&
                                result.get("detection_classes").isPresent() &&
                                result.get("detection_boxes").isPresent()) {

                            //Setup the functions of the model we use
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
                                    System.out.println(imageHeight + " + " + imageWidth);

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
                                            String detectedLabel = cocoLabels[(int) classIndex];
                                            System.out.println("Detected: " + detectedLabel + " with score: " + String.format("%.2f", (score * 100)) + "%.");
                                            returnArray[1] = returnArray[1] + detectedLabel + ": " + String.format("%.2f", (score * 100)) + "%\n";
                                        }
                                    }

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
            return returnArray;
        }
    }

    private int getFileCount(String imagePath) {
        try (Stream<Path> files = Files.walk(Paths.get(imagePath))) {
            return (int) files.filter(Files::isRegularFile).count();
        } catch (IOException e) {
            e.printStackTrace();
            return 0; // Return 0 if there's an error
        }
    }

    public String[] label(String imagePath, SavedModelBundle ModelBundle) { //Add label logic and Json file production!!!

        //Base logic for returning image path and labels
        String[] returnArray = new String[getFileCount(imagePath)];
        System.out.println(returnArray.length);

        for(int i = 0; i < returnArray.length; i++) {

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

                        //Setup the sessions and runners
                        Ops operation = Ops.create(graph);
                        Session.Runner runner = session.runner();

                        //get the file from the file path
                        Operand<TString> filePath = operation.constant(imagePath);
                        ReadFile file = operation.io.readFile(filePath);

                        //Decode the jpeg
                        DecodeJpeg.Options jpeg_options = DecodeJpeg.channels(3L);
                        DecodeJpeg decodeJpeg = operation.image.decodeJpeg(file.contents(), jpeg_options);

                        //Get the shape of the image
                        Shape imageShape = runner.fetch(decodeJpeg).run().get(0).shape();

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

                            //Setup the result operations
                            Result result = ModelBundle.function("serving_default").call(tensorMap);
                            if (result.get("detection_scores").isPresent() &&
                                    result.get("num_detections").isPresent() &&
                                    result.get("detection_classes").isPresent() &&
                                    result.get("detection_boxes").isPresent()) {

                                //Setup the functions of the model we use
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
                                        System.out.println(imageHeight + " + " + imageWidth);

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
                                                System.out.println("Box coordinates: [yMin: " + yMin + ", xMin: " + xMin + ", yMax: " + yMax + ", xMax: " + xMax + "]");

                                                // Get the detected class index and map it to the corresponding label
                                                float classIndex = classes.getFloat(0, j);
                                                String detectedLabel = cocoLabels[(int) classIndex];
                                                System.out.println("Detected: " + detectedLabel + " with score: " + String.format("%.2f", (score * 100)) + "%.");
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
        return returnArray;
    }
}
//Add whole image detection logic!!!
//Add label logic and Json file production!!!