package org.object_d;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.proto.GraphDef;
import org.tensorflow.types.TString;
import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;          


public class Recognizer extends JFrame implements ActionListener {
    JButton predict;
    JTextField result;
    String modelPath;
    byte[] graphDef;
    List<String> labels;

    public Recognizer() {
        setSize(500, 500);
        predict = new JButton("Predict");
        predict.setEnabled(true);
        predict.addActionListener(this);
        result = new JTextField();
        add(result);
        add(predict);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

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
        SwingUtilities.invokeLater(() -> new Recognizer().setVisible(true));
    }

    private static float[] executeInceptionGraph(byte[] graphDef, byte[] imageBytes) {
        try (Graph g = new Graph()) {
            g.importGraphDef(GraphDef.parseFrom(graphDef));
            try (Session s = new Session(g);
                 Tensor image = TString.scalarOf(new String(imageBytes));
                 Tensor result = s.runner()
                         .feed("DecodeJpeg/contents", image)
                         .fetch("softmax")
                         .run().get(0)) {

                long[] rshape = result.shape().asArray();
                int nlabels = (int) rshape[1];
                float[] resultArray = new float[nlabels];
                FloatDataBuffer floatBuffer = result.asRawTensor().data().asFloats();
              //  floatBuffer.copyTo(resultArray);
                return resultArray;
            }
        } catch (Exception e) {
            System.out.println(e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        modelPath = "/Users/gregor/IdeaProjects/U6_CS_TENSORFLOW/";
        graphDef = readAllBytesOrExit(Paths.get(modelPath, "tensorflow_inception_graph.pb"));
        labels = readAllLinesOrExit(Paths.get(modelPath, "imagenet_comp_graph_label_strings.txt"));
        try {
            byte[] imageBytes = Files.readAllBytes(Paths.get("/Users/gregor/IdeaProjects/U6_CS_TENSORFLOW/bild.JPEG"));
            float[] labelProbabilities = executeInceptionGraph(graphDef, imageBytes);
            int bestLabelIdx = maxIndex(labelProbabilities);
            result.setText(String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
        } catch (IOException ex) {
            System.err.println("Failed to read image: " + ex.getMessage());
        }
    }
}
