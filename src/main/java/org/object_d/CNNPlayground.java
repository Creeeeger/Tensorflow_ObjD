package org.object_d;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Stack;

import static org.tensorAction.tensorTrainerCNN.access;

public class CNNPlayground extends JFrame {
    static JButton startButton;
    static JTextArea logTextArea;
    static String filepath;
    static JPanel leftPanel;
    static boolean isValid = true;
    static ArrayList<layerEntry> convertedInputs;

    public CNNPlayground() {
        // Set up the main layout for the frame: 1 row and 2 columns
        setLayout(new GridLayout(1, 2));
        setTitle("CNN Playground"); // Set the title of the window

        // Create the left panel where users can drag and drop blocks
        leftPanel = new JPanel();
        leftPanel.setBorder(BorderFactory.createTitledBorder("Drag & Drop Area - right click for deleting - left click for highlighting and moving"));
        leftPanel.setLayout(new GridLayout(8, 1, 0, 5)); // Arrange components in 8 rows with vertical gaps of 5 pixels
        leftPanel.setBackground(Color.LIGHT_GRAY); // Set background color
        leftPanel.setEnabled(true); // Enable interactions with the panel

        // Create the right panel for actions and block areas
        JPanel rightPanel = new JPanel(new GridLayout(2, 1)); // Split into two vertically stacked areas
        rightPanel.setBorder(BorderFactory.createTitledBorder("Actions Area"));

        // Block Area: Contains buttons and parameters for CNN components
        JPanel blockArea = new JPanel(new GridLayout(3, 2)); // 3 rows, 2 columns
        blockArea.setBorder(BorderFactory.createTitledBorder("Block Area - Drag and drop the blocks below to form a CNN. Set parameters and train the model."));

        // Define the CNN block names and their detailed descriptions
        String[] blockNames = {"Convolutional Layer", "Pooling Layer", "Fully Connected Layer"};
        String[] descriptions = {
                // Explanation for Convolutional Layer
                """
        Convolutional Layer:
        -> goes in windows over the image to find patterns like lines, etc.
        Inputs:
        - Kernel Size: Size of the window? (3 -> a 3x3 square of pixels).
        - Stride: pixels to move window each step? (1 = 1 pixel = small steps).
        - Weight Scale: number for helping the learn process between 0 and 1 (eg. 0.1).
        - Seed: A number to keep the result the same and prevent randomization (eg 1234)""",

                // Explanation for Pooling Layer
                """
        Pooling Layer:
        -> Minimizes the image by reducing the pixels but keeping the most important features
        Inputs:
        - Kernel Size: Size of the window? (2 -> a 2x2 square of pixels).
        - Padding: Should edges have the same size? (Enter 1 for yes, 0 for no).""",

                // Explanation for Fully Connected Layer
                """
        Fully Connected Layer:
        -> makes a final decision about which answer is the right one
        Inputs:
        - Weight Init Scale: initial number for weighting calculations between 0 and 1 (eg. 0.1).
        - Bias Init Value: number for adjusting the answers between 0 and 1 (eg. 0.1).
        - Seed: A number to keep the result the same and prevent randomization (eg 1234)"""
        };

        // Create buttons and panels for each CNN block
        for (int i = 0; i < blockNames.length; i++) {
            JButton infoButton = new JButton(blockNames[i] + " Info"); // Button to display block info

            // Lambda function to display block descriptions in a dialog
            String description = descriptions[i]; // Ensure the correct scope for the lambda
            int finalI = i; // Store the index for reference
            infoButton.addActionListener((ActionEvent _) -> {
                // Display a detailed explanation in a message dialog
                JOptionPane.showMessageDialog(
                        null,
                        "<html><p style='width:600px;'>" + description.replace("\n", "<br>") + "</p></html>",
                        blockNames[finalI] + " Explanation",
                        JOptionPane.INFORMATION_MESSAGE
                );
            });

            // Add the button to the block area
            blockArea.add(infoButton);

            // Retrieve and add the panel specific to this block type
            JPanel blockPanel = getBlockPanel(blockNames[i]); // Function to generate the UI for the block
            blockArea.add(blockPanel); // Add the parameter panel next to the info button
        }

        // Add the block area to the right panel
        rightPanel.add(blockArea);

        // Actions Area: File selection, training controls, and logs
        JPanel actionArea = new JPanel();
        actionArea.setBorder(BorderFactory.createTitledBorder("Actions"));
        actionArea.setLayout(new BoxLayout(actionArea, BoxLayout.Y_AXIS)); // Arrange components vertically

        // Add file selection panel
        JPanel filePanel = addFilePanel(); // Helper function to generate the file selector
        actionArea.add(filePanel);

        // Add the Start Training button
        startButton = new JButton("Start training");
        startButton.setEnabled(false); // Initially disabled until a valid CNN is created
        startButton.addActionListener(_ -> {
            // Validate the CNN configuration before starting training
            if (validateCNN()) {
                logTextArea.append("CNN check passed, start training\n");

                // Debug: Print the converted inputs for each layer
                for (layerEntry convertedInput : convertedInputs) {
                    System.out.println(convertedInput.getLayer() + " " + convertedInput.getList());
                }

                try {
                    // Access and start the training process
                    access(filepath, true, convertedInputs); // Function to handle training logic
                } catch (IOException e) {
                    throw new RuntimeException(e); // Handle training errors
                }

            } else {
                logTextArea.append("Error occurred during CNN validation\n");
            }
        });
        actionArea.add(startButton); // Add the start button to the actions area

        // Log Area: Display logs related to actions
        JPanel logPanel = new JPanel();
        logPanel.setLayout(new BoxLayout(logPanel, BoxLayout.Y_AXIS)); // Stack components vertically

        JLabel logLabel = new JLabel("Logs:"); // Label for the log area
        logTextArea = new JTextArea(10, 30); // Text area for displaying logs
        logTextArea.setEditable(false); // Make the text area read-only
        logTextArea.setText("Logs will appear here...\n"); // Placeholder text
        JScrollPane logScrollPane = new JScrollPane(logTextArea); // Add a scrollable wrapper for the text area

        logPanel.add(logLabel); // Add label to log panel
        logPanel.add(logScrollPane); // Add the scroll pane to the log panel
        actionArea.add(logPanel); // Add the log panel to the actions area

        // Add the actions area to the right panel
        rightPanel.add(actionArea);

        // Enable drag-and-drop functionality for the left panel
        leftPanel.setTransferHandler(new ComponentTransferHandler()); // Custom handler for drag-and-drop

        // Add the left and right panels to the main frame
        add(leftPanel); // Add the drag-and-drop panel
        add(rightPanel); // Add the actions and block configuration panel
    }

    /**
     * Validates the current CNN configuration
     *
     * @return true if the CNN configuration is valid; false otherwise.
     */
    private static boolean validateCNN() {
        // Initialize the validity flag to true
        isValid = true;

        // Stack to collect inputs and block labels in a last-in, first-out manner
        Stack<String> inputsStack = new Stack<>();

        // Recursively process all components in the left panel to populate the stack
        processPanelComponents(leftPanel, inputsStack);

        // Reverse the stack contents using a LinkedList for proper order
        LinkedList<String> collectedInputs = new LinkedList<>();
        while (!inputsStack.isEmpty()) {
            collectedInputs.addFirst(inputsStack.pop());
        }

        // Prepare a list to store structured layer data
        convertedInputs = new ArrayList<>();
        int currentLayerIndex = -1;

        // Iterate through the collected inputs to build structured layers
        for (String input : collectedInputs) {
            if (input.contains("Layer")) {
                // If input contains "Layer", start a new layer entry
                currentLayerIndex++;
                convertedInputs.add(new layerEntry(input, new ArrayList<>()));
            } else {
                try {
                    // Attempt to parse the input as a number and add it to the current layer
                    double number = Double.parseDouble(input);
                    ArrayList<Double> list = convertedInputs.get(currentLayerIndex).getList();
                    list.add(number);
                    convertedInputs.get(currentLayerIndex).setList(list);
                } catch (NumberFormatException e) {
                    // Handle invalid number formats
                    logTextArea.append("Invalid number format: " + input + "\n");
                    isValid = false;
                } catch (IndexOutOfBoundsException e) {
                    // Handle cases where inputs are misplaced or lack a layer
                    logTextArea.append("Error: Input found outside a defined layer: " + input + "\n");
                    isValid = false;
                }
            }
        }

        // Ensure the network is not empty
        if (convertedInputs.isEmpty()) {
            isValid = false;
            logTextArea.append("Can't train an empty network\n");
        }

        return isValid; // Return the validation result
    }

    /**
     * Recursively processes all components within a container
     *
     * @param container   the container with components to be processed
     * @param inputsStack the stack to collect inputs and labels
     */
    private static void processPanelComponents(Container container, Stack<String> inputsStack) {
        // Iterate over all components in the container
        for (Component component : container.getComponents()) {
            if (component instanceof JPanel blockPanel) {
                // If the component is a JPanel, process it as a potential block panel
                processBlockPanel(blockPanel, inputsStack);
            }
        }
    }

    /**
     * extract labels and recursively process child components.
     *
     * @param blockPanel  the JPanel representing a block
     * @param inputsStack the stack to collect inputs and labels
     */
    private static void processBlockPanel(JPanel blockPanel, Stack<String> inputsStack) {
        // Retrieve all components within the block panel
        Component[] blockComponents = blockPanel.getComponents();

        // Ensure the first component is a JLabel representing the block's label
        if (blockComponents.length > 0 && blockComponents[0] instanceof JLabel blockLabel) {
            // Push the block's label text onto the stack
            inputsStack.push(blockLabel.getText());

            // Recursively process child components within the block panel
            processChildComponents(blockPanel, inputsStack);
        }
    }

    /**
     * Processes child components and extract input values.
     *
     * @param blockPanel  the JPanel representing the block
     * @param inputsStack the stack to collect input data
     */
    private static void processChildComponents(JPanel blockPanel, Stack<String> inputsStack) {
        // Iterate over all child components within the block panel
        for (Component child : blockPanel.getComponents()) {
            if (child instanceof JTextField textField) {
                // If the child is a text field, push its value to the stack
                if (textField.getText().contains("Layer")) {
                    // Prevent invalid input that includes "Layer"
                    logTextArea.append("Input text can't contain Layer\n");
                    isValid = false;
                } else {
                    inputsStack.push(textField.getText());
                }
            } else if (child instanceof JPanel nestedPanel) {
                // If a nested JPanel is found, recursively process its components
                processPanelComponents(nestedPanel, inputsStack);
            }
        }
    }

    /**
     * Creates and returns a JPanel for selecting a folder.
     * The panel includes a label, a non-editable text field to display the selected folder path,
     * and a button to open a folder chooser dialog.
     *
     * @return the JPanel containing the file selection UI
     */
    private static JPanel addFilePanel() {
        // Create a new JPanel with a vertical stack layout
        JPanel filePanel = new JPanel();
        filePanel.setLayout(new BoxLayout(filePanel, BoxLayout.Y_AXIS));

        // Label to indicate the purpose of the panel
        JLabel filePathLabel = new JLabel("Folder Path:");

        // TextField to display the selected folder path, initialized as non-editable
        JTextField filePathTextField = new JTextField(30);
        filePathTextField.setEditable(false);  // Prevent user edits
        filePathTextField.setText("No folder selected");  // Placeholder text

        // Button to trigger the folder selection dialog
        JButton fileSelectButton = getFolderPath(filePathTextField);

        // Add components to the panel
        filePanel.add(filePathLabel);
        filePanel.add(filePathTextField);
        filePanel.add(fileSelectButton);

        return filePanel;
    }

    private static JButton getFolderPath(JTextField filePathTextField) {
        JButton fileSelectButton = new JButton("Select Folder");
        fileSelectButton.addActionListener(_ -> {
            JFileChooser folderChooser = new JFileChooser();
            folderChooser.setDialogTitle("Select a Folder");
            folderChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);  // Only allow directories

            // Show the dialog and get the user's choice
            int result = folderChooser.showOpenDialog(null);
            if (result == JFileChooser.APPROVE_OPTION) {
                // Get the selected folder and update the text field
                File selectedFolder = folderChooser.getSelectedFile();
                filePathTextField.setText(selectedFolder.getAbsolutePath());
                startButton.setEnabled(true);  // Enable the start button after folder selection
                filepath = selectedFolder.getAbsolutePath();  // Store the selected path
            }
        });
        return fileSelectButton;
    }

    /**
     * Creates JPanel for specific CNN block types
     * - block-specific parameters
     * Adds drag-and-drop functionality.
     *
     * @param block the name of the CNN block type
     * @return the JPanel representing the block
     */
    private static JPanel getBlockPanel(String block) {
        // Create a new JPanel with a grid layout (variable rows, 2 columns)
        JPanel blockPanel = new JPanel();
        blockPanel.setLayout(new GridLayout(0, 2));  // Flexible row count, 2 columns
        blockPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));  // Black border for visibility
        blockPanel.setOpaque(true);  // Ensure the panel is opaque

        // Add a label for the block name at the top, spanning two columns
        JLabel blockLabel = new JLabel(block, SwingConstants.CENTER);
        blockPanel.add(blockLabel);  // First column
        blockPanel.add(new JLabel());  // Second column as a spacer for alignment

        // Add parameters based on the block type
        switch (block) {
            case "Convolutional Layer" -> {
                blockPanel.add(new JLabel(" Kernel Size:"));  // Label
                blockPanel.add(new JTextField(5));           // Text field for input

                blockPanel.add(new JLabel(" Stride:"));      // Label
                blockPanel.add(new JTextField(5));           // Text field for input

                blockPanel.add(new JLabel(" Weight Scale:"));  // Label
                blockPanel.add(new JTextField(5));            // Text field for input

                blockPanel.add(new JLabel(" Seed:"));        // Label
                blockPanel.add(new JTextField(5));           // Text field for input
            }
            case "Pooling Layer" -> {
                blockPanel.add(new JLabel(" Kernel Size:"));  // Label
                blockPanel.add(new JTextField(5));           // Text field for input

                blockPanel.add(new JLabel(" Padding:"));     // Label
                blockPanel.add(new JTextField(5));           // Text field for input
            }
            case "Fully Connected Layer" -> {
                blockPanel.add(new JLabel(" Weight Init Scale:"));  // Label
                blockPanel.add(new JTextField(5));                  // Text field for input

                blockPanel.add(new JLabel(" Bias Init Value:"));    // Label
                blockPanel.add(new JTextField(5));                  // Text field for input

                blockPanel.add(new JLabel(" Seed:"));              // Label
                blockPanel.add(new JTextField(5));                 // Text field for input
            }
        }

        // Add drag-and-drop functionality to the block panel
        blockPanel.setTransferHandler(new ComponentTransferHandler());
        blockPanel.addMouseListener(new java.awt.event.MouseAdapter() {
            @Override
            public void mousePressed(java.awt.event.MouseEvent evt) {
                // Trigger drag-and-drop functionality on mouse press
                JComponent component = (JComponent) evt.getSource();
                TransferHandler handler = component.getTransferHandler();
                handler.exportAsDrag(component, evt, TransferHandler.MOVE);
            }
        });

        return blockPanel;
    }

    /**
     * Custom TransferHandler for handling drag-and-drop operations with JComponents.
     */
    private static class ComponentTransferHandler extends TransferHandler {

        // Define a custom DataFlavor for JComponent objects
        private static final DataFlavor COMPONENT_FLAVOR = new DataFlavor(JComponent.class, "JComponent");

        /**
         * Recursively prints the layouts of all panels
         *
         * @param component the JComponent to inspect
         */
        private static void printPanelLayoutsRecursively(JComponent component) {
            if (component instanceof JPanel panel) {
                System.out.println("Panel: " + panel.getLayout());
                // Recursively process child components of the panel
                for (Component child : panel.getComponents()) {
                    printPanelLayoutsRecursively((JComponent) child); // Recurse through child components
                }
            }
        }

        /**
         * Creates a Transferable object containing the JComponent being dragged.
         *
         * @param c the JComponent being dragged
         * @return a Transferable object wrapping the JComponent
         */
        @Override
        protected Transferable createTransferable(JComponent c) {
            return new ComponentTransferable(c); // Return a Transferable containing the component
        }

        /**
         * Returns the source actions for the drag operation.
         *
         * @param c the JComponent being dragged
         * @return the source action (MOVE)
         */
        @Override
        public int getSourceActions(JComponent c) {
            return MOVE; // Only allow the MOVE action
        }

        /**
         * Determines if the data can be imported into the drop target.
         *
         * @param support the TransferSupport for the drop operation
         * @return true if the data can be imported, false otherwise
         */
        @Override
        public boolean canImport(TransferSupport support) {
            // Get the drop target component
            Component dropTarget = support.getComponent();

            // Check if the target is a JPanel with a TitledBorder that matches the "Drag & Drop Area" title
            if (dropTarget instanceof JPanel targetPanel) {
                Border border = targetPanel.getBorder();

                // Check if the panel has a TitledBorder
                if (border instanceof TitledBorder) {
                    String panelTitle = ((TitledBorder) border).getTitle();
                    if (panelTitle.contains("Drag & Drop Area")) {
                        return support.isDataFlavorSupported(COMPONENT_FLAVOR); // Check if the correct data flavor is supported
                    } else {
                        return false; // Reject import if title doesn't match
                    }
                }
            }

            return super.canImport(support); // Use the default behavior otherwise
        }

        /**
         * Handles the import of data
         * logic for highlighting, deletion, and dragging the component.
         *
         * @param support the TransferSupport for the drop operation
         * @return true if the component is successfully imported, false otherwise
         */
        @Override
        public boolean importData(TransferSupport support) {
            if (!canImport(support)) {
                return false; // If the drop target is invalid, return false
            }

            try {
                // Get the component being dragged
                JComponent component = (JComponent) support.getTransferable().getTransferData(COMPONENT_FLAVOR);

                // Debug: Print the layout structure of the dragged component
                printPanelLayoutsRecursively(component);

                // Get the target container where the component will be dropped
                Container targetContainer = (Container) support.getComponent();

                // Remove the component from its previous container (if any)
                Container parent = component.getParent();
                if (parent != null) {
                    parent.remove(component);
                }

                // Add mouse listener to handle highlighting, deletion, and dragging
                component.addMouseListener(new java.awt.event.MouseAdapter() {
                    private boolean isHighlighted = false; // Track if the component is highlighted
                    private Point initialClick; // Track initial mouse click for dragging
                    private Point offset; // Track initial position of the component

                    /**
                     * Handles mouse click events for the component.
                     * Right-click deletes the component, left-click toggles highlighting.
                     *
                     * @param evt the mouse event
                     */
                    @Override
                    public void mouseClicked(java.awt.event.MouseEvent evt) {
                        if (SwingUtilities.isRightMouseButton(evt)) {
                            // Right-click: Delete the component
                            targetContainer.remove(component);
                            targetContainer.revalidate();
                            targetContainer.repaint();
                        } else if (SwingUtilities.isLeftMouseButton(evt)) {
                            // Left-click: Toggle highlight color
                            if (isHighlighted) {
                                component.setBackground(Color.DARK_GRAY); // Remove highlight
                                isHighlighted = false;
                            } else {
                                component.setBackground(Color.lightGray); // Highlight the component
                                isHighlighted = true;
                            }
                        }
                    }

                    /**
                     * Handles mouse press events for starting the dragging operation.
                     *
                     * @param evt the mouse event
                     */
                    @Override
                    public void mousePressed(java.awt.event.MouseEvent evt) {
                        if (isHighlighted && SwingUtilities.isLeftMouseButton(evt)) {
                            // Store the initial position and click point for dragging
                            initialClick = evt.getPoint();
                            offset = component.getLocation();
                        }
                    }

                    /**
                     * Handles mouse drag events to move the component.
                     *
                     * @param evt the mouse event
                     */
                    @Override
                    public void mouseDragged(java.awt.event.MouseEvent evt) {
                        // Check if the component is highlighted, an initial click point exists, and the drag is a left-click
                        if (isHighlighted && initialClick != null && SwingUtilities.isLeftMouseButton(evt)) {
                            // Calculate the displacement (dx, dy) from the initial click point to the current mouse position
                            int dx = evt.getX() - initialClick.x;
                            int dy = evt.getY() - initialClick.y;

                            // Compute the new location of the component by adding the displacement to the original offset
                            Point newLocation = new Point(offset.x + dx, offset.y + dy);

                            // Update the component's location to the newly calculated position
                            component.setLocation(newLocation);

                            // Repaint the target container to visually reflect the component's updated position
                            targetContainer.repaint();
                        }
                    }

                    /**
                     * Handles mouse release events to finalize the drop operation.
                     *
                     * @param evt the mouse event
                     */
                    @Override
                    public void mouseReleased(java.awt.event.MouseEvent evt) {
                        if (isHighlighted && SwingUtilities.isLeftMouseButton(evt)) {
                            // Handle drop logic after dragging the component
                            Point currentPoint = evt.getPoint();
                            Point targetPoint = SwingUtilities.convertPoint(component, currentPoint, targetContainer);

                            boolean placedInTarget = false;

                            // Check if the component is dropped within a valid position
                            for (Component c : targetContainer.getComponents()) {
                                if (c != component && c.getBounds().contains(targetPoint)) {
                                    // Reorder the component to the drop position
                                    targetContainer.remove(component);
                                    targetContainer.add(component, targetContainer.getComponentZOrder(c));
                                    targetContainer.revalidate();
                                    targetContainer.repaint();
                                    placedInTarget = true;
                                    break;
                                }
                            }

                            // If no valid position, reset to the original location
                            if (!placedInTarget) {
                                component.setLocation(offset);
                            }
                        }
                    }
                });

                // Initially add the component to the new container (targetContainer)
                targetContainer.add(component);
                targetContainer.revalidate();
                targetContainer.repaint();
                return true; // Return true to indicate the import was successful
            } catch (Exception e) {
                System.out.println(e.getMessage()); // Log any exceptions
                return false; // Return false if an error occurred
            }
        }

        /**
         * Custom Transferable implementation for components.
         */
        private record ComponentTransferable(JComponent component) implements Transferable, Serializable {

            @Override
            public DataFlavor[] getTransferDataFlavors() {
                return new DataFlavor[]{COMPONENT_FLAVOR}; // Return the custom data flavor
            }

            @Override
            public boolean isDataFlavorSupported(DataFlavor flavor) {
                return COMPONENT_FLAVOR.equals(flavor); // Check if the data flavor matches
            }

            @Override
            public Object getTransferData(DataFlavor flavor) {
                return component; // Return the component being transferred
            }
        }
    }

    /**
     * Represents entry for layer
     */
    public static class layerEntry {

        // Name of the layer
        private final String layer;

        // List of numerical parameters associated with the layer
        private ArrayList<Double> list;

        /**
         * Constructs a new layerEntry
         *
         * @param layer the name of the layer
         * @param list  the list of numerical parameters for the layer
         */
        public layerEntry(String layer, ArrayList<Double> list) {
            this.layer = layer; // Set the layer name
            this.list = list;   // Set the parameter list
        }

        /**
         * Gets the name of the layer.
         *
         * @return the name of the layer
         */
        public String getLayer() {
            return layer;
        }

        /**
         * Gets the list of parameters for the layer.
         *
         * @return the list of numerical parameters
         */
        public ArrayList<Double> getList() {
            return list;
        }

        /**
         * Sets the list of numerical parameters for the layer.
         *
         * @param list the new list of numerical parameters
         */
        public void setList(ArrayList<Double> list) {
            this.list = list;
        }

        /**
         * Returns a string representation of object
         *
         * @return a string
         */
        @Override
        public String toString() {
            return "LayerEntry{" +
                    "layer='" + layer + '\'' +
                    ", list=" + Arrays.toString(list.toArray()) +
                    '}';
        }
    }
}