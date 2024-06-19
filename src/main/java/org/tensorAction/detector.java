package org.tensorAction;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;

import java.io.File;
import java.util.TreeMap;

public class detector {
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

    public String[] classify(String imagePath, SavedModelBundle ModelBundle) {

        //Base logic for returning image path and labels
        String path = "";
        String result = "";
        String[] returnArray = new String[2];
        returnArray[0] = path;
        returnArray[1] = result;

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

            for (int i = 0; i < cocoLabels.length; i++) {
                cocoLabelMap.put(Label_count, cocoLabels[i]);
                Label_count++;
            }
        }

        //Setup graph and session and operation graph
        try (Graph graph = new Graph()) {
            try (Session session = new Session(graph)) {
                Ops operation = Ops.create(graph);

            }
        }

        return returnArray;
    }
}
//Add whole image detection logic!!!
