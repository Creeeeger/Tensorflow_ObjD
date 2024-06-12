import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.proto.GraphDef;
import org.tensorflow.types.TString;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Recognizer {
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    public static void main(String[] args) {
        execute();
    }

    private static float[][] getResultArray(Tensor result) {
        final long[] rshape = result.shape().asArray();
        if (result.shape().numDimensions() != 2 || rshape[0] != 1) {
            throw new RuntimeException(
                    String.format(
                            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                            Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return new float[1][nlabels];
    }

    public static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            try {
                g.importGraphDef(GraphDef.parseFrom(graphDef));
            } catch (InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
            }
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                float[][] resultArray = getResultArray(result);
                //result.copyTo(resultArray);
                return resultArray[0];
            }
        }
    }

    public static void execute() {
        String modelPath = "/Users/gregor/Desktop/Tensorflow_ObjD/";
        byte[] graphDef = readAllBytesOrExit(Paths.get(modelPath, "tensorflow_inception_graph.pb"));
        List<String> labels = readAllLinesOrExit(Paths.get(modelPath, "imagenet_comp_graph_label_strings.txt"));
        try {

            byte[] imageBytes = Files.readAllBytes(Paths.get("/Users/gregor/Desktop/Tensorflow_ObjD/bild.JPEG"));
            NdArray byteNdArray = NdArrays.ofBytes(org.tensorflow.ndarray.Shape.of(imageBytes.length));
            TString tensor = TString.tensorOfBytes(byteNdArray);

            float[] labelProbabilities = executeInceptionGraph(graphDef, tensor);
            int bestLabelIdx = maxIndex(labelProbabilities);
            System.out.printf("BEST MATCH: %s (%.2f%% likely)%n", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f);
        } catch (IOException ex) {
            System.err.println("Failed to read image: " + ex.getMessage());
        }
    }
}
