package org.object_d;

import org.tensorflow.*;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.proto.DataType;
import org.tensorflow.types.TFloat32;

import java.io.IOException;
import java.nio.file.Paths;

// Class that demonstrates saving a TensorFlow model with a placeholder and a constant
public class StageTwoExporter {
    // Main method: entry point of the application
    public static void main(String[] args) throws IOException {
        saveGraphToDisk(Paths.get(Paths.get(Paths.get("").toAbsolutePath().toString()).getParent().toString()).toString(), 20);
    }

    /**
     * Creates a TensorFlow graph with a constant, a placeholder, and an addition operation.
     *
     * @param modelPath Path where the model will be saved
     * @param intVal    An integer value to test the model during inference
     */
    public static void saveGraphToDisk(String modelPath, int intVal) throws IOException {
        // Try-with-resources to ensure the graph is properly closed after use
        try (Graph graph = new Graph()) {
            // Define the constant tensor with a value of 1 as TFloat32
            try (Tensor tensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[]{1}))) {
                // Create a constant operation and add it to the graph
                Output<TFloat32> constantTensor = graph.opBuilder("Const", "v", graph.baseScope())
                        .setAttr("dtype", DataType.DT_FLOAT) // Set the data type to TFloat32 (float)
                        .setAttr("value", tensor) // Assign the tensor value (1.0)
                        .build() // Build the operation
                        .output(0); // Get the output of the operation

                // Define a placeholder operation to accept a dynamic input during inference
                Operand<TFloat32> inputInteger = graph.opBuilder("Placeholder", "input", graph.baseScope())
                        .setAttr("dtype", DataType.DT_FLOAT) // Set the data type for the placeholder to TFloat32
                        .build() // Build the placeholder operation
                        .output(0); // Get the output of the operation

                // Add the constant tensor to the input placeholder (addition operation)
                Operand<TFloat32> addedValue = graph.opBuilder("Add", "add", graph.baseScope())
                        .addInput(constantTensor) // Input: the constant tensor
                        .addInput(inputInteger.asOutput()) // Input: the placeholder value
                        .build() // Build the addition operation
                        .output(0); // Get the output of the operation

                // Create a SessionFunction for managing the session and operations in the graph
                SessionFunction function = new SessionFunction(
                        new Signature.Builder().build(), // Define an empty signature for this example
                        new Session(graph) // Create a session for the graph
                ).withNewSession(new Session(graph)); // Add a new session to the function

                // Save the graph as a TensorFlow SavedModel
                SavedModelBundle.exporter(modelPath) // Specify the export location
                        .withTags("serve") // Set the "serve" tag for inference purposes
                        .withFunction(function) // Include the session function in the model
                        .export(); // Save the model to disk

                // Call the detect method to test the saved model with the provided input
                detect(modelPath, intVal);
            }
        }
    }

    /**
     * Loads the saved TensorFlow model and performs inference
     *
     * @param modelPath  Path to the saved TensorFlow model
     * @param inputValue The integer input value for the model's placeholder
     */
    public static void detect(String modelPath, int inputValue) {
        // Load the saved model from the specified path
        try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
            // Create a session for running the model
            try (Session session = model.session()) {
                // Convert the input integer value to a TFloat32 tensor
                try (Tensor inputTensor = TFloat32.scalarOf((float) inputValue)) { // Create a scalar tensor from inputValue
                    // Run the session with the input tensor and fetch the result of the addition
                    TFloat32 result = (TFloat32) session.runner()
                            .feed("input", inputTensor) // Feed the input tensor to the placeholder
                            .fetch("add") // Fetch the output of the "add" operation
                            .run() // Execute the session
                            .get(0); // Retrieve the first result (TFloat32 tensor)

                    // Print the fetched value (result of constant + inputValue)
                    System.out.println("Result: " + result.getFloat());
                }
            }
        }
    }
}