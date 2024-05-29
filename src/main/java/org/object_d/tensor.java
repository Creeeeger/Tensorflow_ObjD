package org.object_d;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.TensorFlow;

public class tensor {
    public static void main(String[] args) {
        System.out.println(TensorFlow.version());
    }

    public static void load_model(String path) {
        try {
            SavedModelBundle model = SavedModelBundle.load(path, "serve");
            // Now the model is loaded and ready to use
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
