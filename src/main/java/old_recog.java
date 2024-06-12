import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.proto.GraphDef;
import org.tensorflow.types.TString;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class old_recog extends JFrame implements ActionListener {
    private final JButton predict;
    private final JButton incep;
    private final JButton img;
    private final JFileChooser incepch;
    private final JFileChooser imgch;
    private final JLabel viewer;
    private final JTextField result;
    private final JTextField imgpth;
    private final JTextField modelpth;
    private String imagepath;
    private boolean modelselected = false;
    private byte[] graphDef;
    private List<String> labels;

    public old_recog() {
        setLayout(new GridLayout(4, 4));
        setSize(500, 500);

        predict = new JButton("Predict");
        predict.setEnabled(false);
        incep = new JButton("Choose Inception");
        img = new JButton("Choose Image");
        incep.addActionListener(this);
        img.addActionListener(this);
        predict.addActionListener(this);

        incepch = new JFileChooser();
        imgch = new JFileChooser();
        FileNameExtensionFilter imgfilter = new FileNameExtensionFilter("JPG & JPEG Images", "jpg", "jpeg");
        imgch.setFileFilter(imgfilter);
        imgch.setFileSelectionMode(JFileChooser.FILES_ONLY);
        incepch.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        result = new JTextField();
        modelpth = new JTextField();
        imgpth = new JTextField();
        modelpth.setEditable(false);
        imgpth.setEditable(false);
        viewer = new JLabel();
        add(modelpth);
        add(incep);
        add(imgpth);
        add(img);
        add(viewer).setSize(new Dimension(200, 200));
        add(predict);
        add(result).setSize(new Dimension(300, 100));
        setLocationRelativeTo(null);
        setResizable(false);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
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
                final long[] rshape = result.shape().asArray();
                if (result.shape().numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                float[][] resultArray = new float[1][nlabels];
                // result.copyTo(resultArray);
                return resultArray[0];
            }
        }
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
            return Files.readAllLines(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new old_recog().setVisible(true));
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        if (e.getSource() == incep) {
            int returnVal = incepch.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = incepch.getSelectedFile();
                String modelpath = file.getAbsolutePath();
                modelpth.setText(modelpath);
                System.out.println("Opening: " + file.getAbsolutePath());
                modelselected = true;
                graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
                labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));
            } else {
                System.out.println("Process was cancelled by user.");
            }

        } else if (e.getSource() == img) {
            int returnVal = imgch.showOpenDialog(old_recog.this);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = imgch.getSelectedFile();
                    imagepath = file.getAbsolutePath();
                    imgpth.setText(imagepath);
                    System.out.println("Image Path: " + imagepath);
                    Image img = ImageIO.read(file);

                    viewer.setIcon(new ImageIcon(img.getScaledInstance(200, 200, 200)));
                    if (modelselected) {
                        predict.setEnabled(true);
                    }
                } catch (IOException ex) {
                    Logger.getLogger(old_recog.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                System.out.println("Process was cancelled by user.");
            }
        } else if (e.getSource() == predict) {
            try {
                byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));
                NdArray ndArray = NdArrays.vectorOf(imageBytes);
                TString tensor = TString.tensorOfBytes(ndArray);

                float[] labelProbabilities = executeInceptionGraph(graphDef, tensor);
                int bestLabelIdx = maxIndex(labelProbabilities);
                result.setText("");
                result.setText(String.format(
                        "BEST MATCH: %s (%.2f%% likely)",
                        labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
                System.out.printf(
                        "BEST MATCH: %s (%.2f%% likely)%n",
                        labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f);
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}