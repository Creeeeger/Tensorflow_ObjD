package org.tensorflow.model.examples.cnn.fastrcnn;

import org.tensorflow.*;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.EncodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.op.io.WriteFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class FasterRcnnInception {
    private final static String[] cocoLabels = new String[]{
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "street sign",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "hat",
            "backpack",
            "umbrella",
            "shoe",
            "eye glasses",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "plate",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "mirror",
            "dining table",
            "window",
            "desk",
            "toilet",
            "door",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "blender",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
            "hair brush"
    };

    public static void main(String[] params) {
        String outputImagePath = "/Users/gregor/Desktop/output.jpeg";
        String imagePath = "/Users/gregor/Desktop/Tensorflow_ObjD/car.jpg";
        String modelPath = "/Users/gregor/Desktop/Tensorflow_ObjD/src/main/resources/model";
        try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
            //create a map of the COCO 2017 labels
            TreeMap<Float, String> cocoTreeMap = new TreeMap<>();
            float cocoCount = 0;
            for (String cocoLabel : cocoLabels) {
                cocoTreeMap.put(cocoCount, cocoLabel);
                cocoCount++;
            }
            try (Graph g = new Graph(); Session s = new Session(g)) {
                Ops tf = Ops.create(g);
                Constant<TString> fileName = tf.constant(imagePath);
                ReadFile readFile = tf.io.readFile(fileName);
                Session.Runner runner = s.runner();
                DecodeJpeg.Options options = DecodeJpeg.channels(3L);
                DecodeJpeg decodeImage = tf.image.decodeJpeg(readFile.contents(), options);
                //fetch image from file
                Shape imageShape = runner.fetch(decodeImage).run().get(0).shape();
                //reshape the tensor to 4D for input to model
                Reshape<TUint8> reshape = tf.reshape(decodeImage,
                        tf.array(1,
                                imageShape.asArray()[0],
                                imageShape.asArray()[1],
                                imageShape.asArray()[2]
                        )
                );
                try (TUint8 reshapeTensor = (TUint8) s.runner().fetch(reshape).run().get(0)) {
                    Map<String, Tensor> feedDict = new HashMap<>();
                    //The given SavedModel SignatureDef input
                    feedDict.put("input_tensor", reshapeTensor);
                    //The given SavedModel MetaGraphDef key
                    Result outputTensorMap = model.function("serving_default").call(feedDict);
                    //detection_classes, detectionBoxes etc. are model output names
                    try (TFloat32 detectionClasses = (TFloat32) outputTensorMap.get("detection_classes").get();
                         TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes").get();
                         TFloat32 numDetections = (TFloat32) outputTensorMap.get("num_detections").get();
                         TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores").get()
                         //;
                         //TFloat32 rawDetectionBoxes = (TFloat32) outputTensorMap.get("raw_detection_boxes").get();
                         //TFloat32 rawDetectionScores = (TFloat32) outputTensorMap.get("raw_detection_scores").get();
                         //TFloat32 detectionAnchorIndices = (TFloat32) outputTensorMap.get("detection_anchor_indices").get();
                         //TFloat32 detectionMulticlassScores = (TFloat32) outputTensorMap.get("detection_multiclass_scores").get()
                    ) {
                        int numDetects = (int) numDetections.getFloat(0);
                        if (numDetects > 0) {
                            ArrayList<FloatNdArray> boxArray = new ArrayList<>();
                            //TODO tf.image.combinedNonMaxSuppression
                            for (int n = 0; n < numDetects; n++) {
                                //put probability and position in outputMap
                                float detectionScore = detectionScores.getFloat(0, n);
                                //only include those classes with detection score greater than 0.3f
                                if (detectionScore > 0.3f) {
                                    boxArray.add(detectionBoxes.get(0, n));
                                    // Get the detected class index and map it to the corresponding label
                                    float detectedClassIndex = detectionClasses.getFloat(0, n);
                                    String detectedLabel = cocoLabels[(int) detectedClassIndex];
                                    System.out.println("Detected: " + detectedLabel + " with score: " + String.format("%.2f", (detectionScore * 100)) + "%.");
                                }
                            }
                            //2-D. A list of RGBA colors to cycle through for the boxes.
                            Operand<TFloat32> colors = tf.constant(new float[][]{
                                    {0.9f, 0.3f, 0.3f, 0.0f},
                                    {0.3f, 0.3f, 0.9f, 0.0f},
                                    {0.3f, 0.9f, 0.3f, 0.0f}
                            });
                            Shape boxesShape = Shape.of(1, boxArray.size(), 4);
                            //3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes
                            try (TFloat32 boxes = TFloat32.tensorOf(boxesShape)) {
                                // batch size of 1
                                for (int i = 0; i < boxArray.size(); i++) {
                                    boxes.set(boxArray.get(i), 0, i);
                                }
                                //Placeholders for boxes and path to output image
                                Placeholder<TFloat32> boxesPlaceHolder = tf.placeholder(TFloat32.class, Placeholder.shape(boxesShape));
                                Placeholder<TString> outImagePathPlaceholder = tf.placeholder(TString.class);
                                //Create JPEG from the Tensor with quality of 100%
                                EncodeJpeg.Options jpgOptions = EncodeJpeg.quality(100L);
                                //convert the 4D input image to normalised 0.0f - 1.0f
                                //Draw bounding boxes using boxes tensor and list of colors
                                //multiply by 255 then reshape and recast to TUint8 3D tensor
                                WriteFile writeFile = tf.io.writeFile(outImagePathPlaceholder,
                                        tf.image.encodeJpeg(
                                                tf.dtypes.cast(tf.reshape(
                                                        tf.math.mul(
                                                                tf.image.drawBoundingBoxes(tf.math.div(
                                                                                tf.dtypes.cast(tf.constant(reshapeTensor), TFloat32.class), tf.constant(255.0f)),
                                                                        boxesPlaceHolder, colors),
                                                                tf.constant(255.0f)
                                                        ),
                                                        tf.array(
                                                                imageShape.asArray()[0],
                                                                imageShape.asArray()[1],
                                                                imageShape.asArray()[2]
                                                        )
                                                ), TUint8.class),
                                                jpgOptions));
                                //output the JPEG to file
                                s.runner().feed(outImagePathPlaceholder, TString.scalarOf(outputImagePath))
                                        .feed(boxesPlaceHolder, boxes)
                                        .addTarget(writeFile).run();
                            }
                        }
                    }
                }
            }
        }
    }
}