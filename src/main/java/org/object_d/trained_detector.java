package org.object_d;

import org.tensorflow.*;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.MaxPool;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class trained_detector extends JFrame {

    //Initialise the elements
    public static File tensor_file = new File("/");
    static JLabel Tensor_name, image_name, output_name, img;
    static JButton image_select, tensor_select, predict;
    static SavedModelBundle savedModelBundle;

    public trained_detector() {
        //Create the layout
        setLayout(new BorderLayout(10, 10)); // Use BorderLayout with spacing
        JPanel detectorPanel = new JPanel();
        detectorPanel.setLayout(new BoxLayout(detectorPanel, BoxLayout.Y_AXIS));
        detectorPanel.setBorder(BorderFactory.createTitledBorder("Detector from previously created models")); // Add border with title

        //Set up the components
        Tensor_name = new JLabel("Tensor file");
        tensor_select = new JButton("Select Tensor file");

        image_name = new JLabel("Image file");
        image_select = new JButton("Select image file");
        image_select.setEnabled(false);

        output_name = new JLabel("Predicted class");

        predict = new JButton("Predict");
        predict.setEnabled(false);

        // Create a dummy image placeholder
        BufferedImage placeholderImage = new BufferedImage(200, 200, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = placeholderImage.createGraphics();
        g2d.setColor(Color.GRAY);
        g2d.fillRect(0, 0, 200, 200);
        g2d.setColor(Color.BLACK);
        g2d.drawString("Image comes here", 50, 100);
        g2d.dispose();
        ImageIcon dummyImage = new ImageIcon(placeholderImage);
        img = new JLabel(dummyImage);

        // Add space between components
        detectorPanel.add(Tensor_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(tensor_select);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(image_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(image_select);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(img);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(predict);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        detectorPanel.add(output_name);
        detectorPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        //Add the button actions
        image_select.addActionListener(new event_select_image(img));
        tensor_select.addActionListener(new event_select_tensor());
        predict.addActionListener(new event_predict());

        add(detectorPanel, BorderLayout.CENTER);
    }

    public static void main(String[] args) {
        trained_detector gui = new trained_detector();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setVisible(true);
        gui.setTitle("Object detector for own models");
        gui.setSize(600, 600);
    }

    public static class event_select_image implements ActionListener {
        JLabel imageLabel;

        public event_select_image(JLabel imageLabel) {
            this.imageLabel = imageLabel;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();
                try {
                    ImageIcon icon = new ImageIcon(selectedFile.getPath());
                    Image originalImage = icon.getImage();
                    int desiredHeight = 300;
                    int desiredWidth = 400;
                    Image scaledImage = originalImage.getScaledInstance(desiredWidth, desiredHeight, Image.SCALE_SMOOTH);
                    ImageIcon scaledIcon = new ImageIcon(scaledImage);
                    imageLabel.setIcon(scaledIcon);
                    image_name.setText(selectedFile.getPath());
                    predict.setEnabled(true);

                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }
    }

    public static class event_select_tensor implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showOpenDialog(null);
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                tensor_file = fileChooser.getSelectedFile();
                Tensor_name.setText(tensor_file.getPath());
            }

            try {
                savedModelBundle = SavedModelBundle.load(tensor_file.getPath(), "serve");
                System.out.println("Model loaded");
                image_select.setEnabled(true);

            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static class event_predict implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            detect();
        }
    }

    public static void detect() {
        //Setup graph and session and operation graph
        try (Graph graph = new Graph()) {
            try (Session session = new Session(graph)) {
                Ops operation = Ops.create(graph);

                //Decode the jpeg and get the file
                DecodeJpeg decodeJpeg = operation.image.decodeJpeg(operation.io.readFile(operation.constant(image_name.getText())).contents(), DecodeJpeg.channels(3L));

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
                    tensorMap.put("input", reshape_Tensor);
                    try (Result result = savedModelBundle.function("serving_default").call(tensorMap)) {
                        try(TFloat32 amount = (TFloat32) result.get("output").get()) {
                            float detections = amount.getFloat(0);
                            System.out.println(detections);
                        }

                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            }
        }
    }
}