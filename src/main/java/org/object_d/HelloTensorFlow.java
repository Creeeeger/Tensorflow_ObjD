package org.object_d;

import org.tensorflow.TensorFlow;

public class HelloTensorFlow {
    public String version() {
        return TensorFlow.version();
    }
}