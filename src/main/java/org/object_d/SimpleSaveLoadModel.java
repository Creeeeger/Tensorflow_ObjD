package org.object_d;

import org.tensorflow.*;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.proto.DataType;
import org.tensorflow.types.TFloat32;

import java.io.IOException;
import java.nio.file.Paths;

// Class that demonstrates saving and loading a TensorFlow model
public class SimpleSaveLoadModel {
    // Main method: entry point of the application
    public static void main(String[] args) throws IOException {
        // Call the method to create, save, and load the TensorFlow model
        saveGraphToDisk(Paths.get(Paths.get(Paths.get("").toAbsolutePath().toString()).getParent().toString()).toString());
    }

    /**
     * Creates a simple TensorFlow graph, saves it to disk as a model
     *
     * @param modelPath Path to save the TensorFlow model
     */
    public static void saveGraphToDisk(String modelPath) {
        // Try-with-resources block to ensure the Graph is properly closed
        try (Graph graph = new Graph()) {
            // Define a variable to hold the SessionFunction for the model
            SessionFunction function;

            // Create a TensorFlow tensor containing a single float value (4.0)
            try (Tensor tensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[]{4}))) {
                // Add a constant tensor operation to the graph
                graph.opBuilder("Const", "v", graph.baseScope()) // Create a "Const" operation named "v"
                        .setAttr("dtype", DataType.DT_FLOAT) // Set the data type of the tensor (float)
                        .setAttr("value", tensor) // Assign the tensor as the value of this operation
                        .build(); // Build the operation into the graph

                // Create a new session function for the graph
                function = new SessionFunction(new Signature.Builder().build(), new Session(graph))
                        .withNewSession(new Session(graph));
            }

            // Save the constructed graph as a SavedModel to disk
            try {
                // Export the model as a "serve" tag for inference purposes
                SavedModelBundle.exporter(modelPath)
                        .withTags("serve") // Define the tag for serving
                        .withFunction(function) // Include the session function in the model
                        .export(); // Save the model to the specified path

                // Call the detect method to perform inference using the saved model
                detect(modelPath);
            } catch (IOException e) {
                // Rethrow IOException as a RuntimeException
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Loads a saved TensorFlow model from disk and fetch a constant value.
     *
     * @param modelPath Path where the TensorFlow model is saved
     */
    public static void detect(String modelPath) {
        // Load the saved model from the specified path
        try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
            // Open a new session for inference
            try (Session session = model.session()) {
                // Fetch the value of the constant tensor "v" from the graph
                TFloat32 result = (TFloat32) session.runner()
                        .fetch("v") // Specify the name of the tensor to fetch
                        .run() // Execute the session and fetch results
                        .get(0); // Retrieve the first output tensor

                // Print the fetched value to the console
                System.out.println("Fetched value from the model: " + result.getFloat());
            }
        }
    }
}