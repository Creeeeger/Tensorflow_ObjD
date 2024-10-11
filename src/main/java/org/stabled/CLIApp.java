package org.stabled;

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.logging.Logger;

public final class CLIApp {
    private static final Logger logger = Logger.getLogger(CLIApp.class.getName());

    private CLIApp() {
        // Private constructor to prevent instantiation of the utility class
    }

    public static void gen(String[] args, int steps, String prompt, int B_size, String output_path) throws OrtException, IOException {
        // Generates an image based on a given prompt using SD4J

        // Parse command line arguments to create an SD4JConfig (specific configuration for Stable Diffusion model usage)
        Optional<SD4J.SD4JConfig> config = SD4J.SD4JConfig.parseArgs(args);

        if (config.isEmpty()) {
            // If configuration cannot be created (likely due to missing arguments), print the help message and exit
            System.out.println(SD4J.SD4JConfig.help());
            System.exit(1); // Exit with status code 1 (indicating an error)
        }

        // Create an instance of SD4J using the configuration from the arguments
        SD4J sd = SD4J.factory(config.get());

        // Generate a random seed for generating images, adding some randomness to the output
        int seed = (int) (Math.random() * 1000) + 2;

        // Generate images based on the provided prompt and other parameters
        List<SD4J.SDImage> images = sd.generateImage(
                steps,               // Number of steps to generate the image
                prompt,              // Prompt text for generating the image
                "",                  // Additional conditioning text (empty in this case)
                7.5f,                // Guidance scale, which affects how strongly the model follows the prompt
                B_size,              // Batch size (number of images to generate simultaneously)
                new SD4J.ImageSize(512, 512), // Dimensions of the generated images (512x512 pixels)
                seed                 // Random seed for reproducibility
        );

        // Construct the output path, using the random seed to differentiate file names
        String output = output_path + "/output-" + seed + ".png";

        // Log a message indicating where the image will be saved
        logger.info("Saving to " + output);

        // Save the first generated image in the specified output path
        SD4J.save(images.getFirst(), output);

        // Close the SD4J instance to release resources
        sd.close();
    }
}