package org.object_d;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

public class PayloadHandler {
    private static final String WEBUI_SERVER_URL = "http://127.0.0.1:7860";
    private static final String OUT_DIR_T2I = "api_out/txt2img";

    public static void payloadExecute() {
        try {
            // Ensure output directory exists
            new File(OUT_DIR_T2I).mkdirs();

            // Example payload for txt2img
            String txt2imgPayload = """
                        {
                            "prompt": "masterpiece, (best quality:1.1), 1girl <lora:lora_model:1>",
                            "negative_prompt": "",
                            "seed": 1,
                            "steps": 20,
                            "width": 512,
                            "height": 512,
                            "cfg_scale": 7,
                            "sampler_name": "DPM++ 2M",
                            "n_iter": 1,
                            "batch_size": 1
                        }
                    """;

            // Call txt2img API
            callTxt2ImgApi(txt2imgPayload);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void callTxt2ImgApi(String payload) throws IOException {
        URL url = new URL(WEBUI_SERVER_URL + "/sdapi/v1/txt2img");
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setDoOutput(true);

        try (OutputStream os = connection.getOutputStream()) {
            byte[] input = payload.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        int responseCode = connection.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }

                // Parse response JSON and save images
                saveImagesFromResponse(response.toString());
            }
        } else {
            throw new IOException("API call failed with response code: " + responseCode);
        }
    }

    private static void saveImagesFromResponse(String jsonResponse) {
        // A simple way to parse JSON and extract the images array
        int imagesStartIndex = jsonResponse.indexOf("\"images\":[") + 10;
        int imagesEndIndex = jsonResponse.indexOf("]", imagesStartIndex);
        String imagesArray = jsonResponse.substring(imagesStartIndex, imagesEndIndex);

        // Split base64 encoded images
        String[] images = imagesArray.split(",");

        for (int i = 0; i < images.length; i++) {
            String base64Image = images[i].replace("\"", "");
            saveBase64Image(base64Image, "image-" + System.currentTimeMillis() + "-" + i + ".png");
        }
    }

    private static void saveBase64Image(String base64Image, String fileName) {
        byte[] imageBytes = Base64.getDecoder().decode(base64Image);
        try (FileOutputStream fos = new FileOutputStream(new File(OUT_DIR_T2I, fileName))) {
            fos.write(imageBytes);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}