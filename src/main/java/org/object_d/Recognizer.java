package org.object_d;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
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
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor image = Tensor.create(imageBytes);
                 Tensor result = s.runner()
                         .feed("DecodeJpeg/contents", image)
                         .fetch("softmax")
                         .run().getFirst()) {

                long[] rshape = Arrays.stream(result.shape()).toArray();
                int nlabels = (int) rshape[1];
                float[] resultArray = new float[nlabels];
                float[] floatBuffer = result.copyTo(resultArray);
                return resultArray;
            }
        } catch (Exception e) {
            System.out.println(e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        modelPath = "C:\\Users\\grego\\Desktop\\Tensorflow_ObjD\\";
        graphDef = readAllBytesOrExit(Paths.get(modelPath, "tensorflow_inception_graph.pb"));
        labels = readAllLinesOrExit(Paths.get(modelPath, "imagenet_comp_graph_label_strings.txt"));
        try {
            byte[] imageBytes = Files.readAllBytes(Paths.get("C:\\Users\\grego\\Desktop\\Tensorflow_ObjD\\bild.JPEG"));
            float[] labelProbabilities = executeInceptionGraph(graphDef, imageBytes);
            int bestLabelIdx = maxIndex(labelProbabilities);
            result.setText(String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
        } catch (IOException ex) {
            System.err.println("Failed to read image: " + ex.getMessage());
        }
    }
}
