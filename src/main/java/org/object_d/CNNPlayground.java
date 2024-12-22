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
        setLayout(new GridLayout(1, 2));
        setTitle("CNN Playground");

        leftPanel = new JPanel();
        leftPanel.setBorder(BorderFactory.createTitledBorder("Drag & Drop Area - right click for deleting - left click for highlighting and moving"));
        leftPanel.setLayout(new GridLayout(8, 1, 0, 5)); // Vertical gap of 10 pixels
        leftPanel.setBackground(Color.LIGHT_GRAY);
        leftPanel.setEnabled(true);

        // Right Panel: Actions and Block Area
        JPanel rightPanel = new JPanel(new GridLayout(2, 1));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Actions Area"));

        // Block Area
        JPanel blockArea = new JPanel(new GridLayout(3, 2));
        blockArea.setBorder(BorderFactory.createTitledBorder("Block Area - Drag and drop the blocks below to form a CNN. Set parameters and train the model."));

        // Add draggable blocks
        String[] blockNames = {"Convolutional Layer", "Pooling Layer", "Fully Connected Layer"};
        String[] descriptions = {
                """
Convolutional Layer:
-> goes in windows over the image to find patterns like lines, etc.
Inputs:
- Kernel Size: Size of the window? (3 -> a 3x3 square of pixels).
- Stride: pixels to move window each step? (1 = 1 pixel = small steps).
- Weight Scale: number for helping the learn process between 0 and 1 (eg. 0.1).
- Seed: A number to keep the result the same and prevent randomization (eg 1234)""",

                """
Pooling Layer:
-> Minimizes the image by reducing the pixels but keeping the most important features
Inputs:
- Kernel Size: Size of the window? (2 -> a 2x2 square of pixels).
- Padding: Should edges have the same size? (Enter 1 for yes, 0 for no).""",

                """
Fully Connected Layer:
-> makes a final decision about which answer is the right one\s
Inputs:
- Weight Init Scale: initial number for weighting calculations between 0 and 1 (eg. 0.1).
- Bias Init Value: number for adjusting the answers between 0 and 1 (eg. 0.1).
- Seed: A number to keep the result the same and prevent randomization (eg 1234)"""
        };

        for (int i = 0; i < blockNames.length; i++) {
            // Create a button for each layer
            JButton infoButton = new JButton(blockNames[i] + " Info");

            // Add action listener to display explanation in a new window
            String description = descriptions[i]; // Needed because of lambda scope
            int finalI = i;
            infoButton.addActionListener((ActionEvent _) -> {
                // Show a dialog with the explanation
                JOptionPane.showMessageDialog(
                        null,
                        "<html><p style='width:600px;'>" + description.replace("\n", "<br>") + "</p></html>",
                        blockNames[finalI] + " Explanation",
                        JOptionPane.INFORMATION_MESSAGE
                );
            });

            // Add the button to the block area
            blockArea.add(infoButton);

            // Get the block-specific UI panel
            JPanel blockPanel = getBlockPanel(blockNames[i]);
            blockArea.add(blockPanel); // Add the block panel to the second column
        }

        rightPanel.add(blockArea);

        // Actions Area
        JPanel actionArea = new JPanel();
        actionArea.setBorder(BorderFactory.createTitledBorder("Actions"));
        actionArea.setLayout(new BoxLayout(actionArea, BoxLayout.Y_AXIS)); // Set layout for vertical arrangement

        // File Selector (Folder Selection)
        JPanel filePanel = addFilePanel();
        actionArea.add(filePanel);

        // Start Button
        startButton = new JButton("Start training");
        startButton.setEnabled(false);
        startButton.addActionListener(_ -> {
            // Action for starting the process
            if (validateCNN()) {
                logTextArea.append("CNN check passed, start training\n");

                for (layerEntry convertedInput : convertedInputs) {
                    System.out.println(convertedInput.getLayer() + " " + convertedInput.getList()); //debug layers
                }

                try {
                    access(filepath, true, convertedInputs); // add training access method
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            } else {
                logTextArea.append("Error occurred during CNN validation\n");
            }
        });
        actionArea.add(startButton);

        // Log Text Area
        JPanel logPanel = new JPanel();
        logPanel.setLayout(new BoxLayout(logPanel, BoxLayout.Y_AXIS)); // Stack vertically

        JLabel logLabel = new JLabel("Logs:");  // Label for the log area
        logTextArea = new JTextArea(10, 30);  // 10 rows and 30 columns
        logTextArea.setEditable(false);  // Make the text area read-only
        logTextArea.setText("Logs will appear here...\n");  // Placeholder text
        JScrollPane logScrollPane = new JScrollPane(logTextArea); // Add scroll pane to the text area

        logPanel.add(logLabel);
        logPanel.add(logScrollPane);
        actionArea.add(logPanel);
        rightPanel.add(actionArea);

        // Set TransferHandler for the left panel (Drop Area)
        leftPanel.setTransferHandler(new ComponentTransferHandler());

        // Add Panels to Frame
        add(leftPanel);
        add(rightPanel);
    }

    // more Recursion
    private static boolean validateCNN() {
        // Initialize the validity flag
        isValid = true;

        // Stack to collect inputs and block labels
        Stack<String> inputsStack = new Stack<>();

        // Recursively process the left panel and populate the stack
        processPanelComponents(leftPanel, inputsStack);

        // Use a LinkedList to efficiently reverse the stack contents
        LinkedList<String> collectedInputs = new LinkedList<>();
        while (!inputsStack.isEmpty()) {
            collectedInputs.addFirst(inputsStack.pop());
        }

        // Convert collected inputs into structured layers
        convertedInputs = new ArrayList<>();
        int currentLayerIndex = -1;

        for (String input : collectedInputs) {
            if (input.contains("Layer")) {
                // Start a new layer if the input contains "Layer"
                currentLayerIndex++;
                convertedInputs.add(new layerEntry(input, new ArrayList<>()));
            } else {
                try {
                    // Parse the input as a number and add to the current layer
                    double number = Double.parseDouble(input);
                    ArrayList<Double> list = convertedInputs.get(currentLayerIndex).getList();
                    list.add(number);
                    convertedInputs.get(currentLayerIndex).setList(list);
                } catch (NumberFormatException e) {
                    // Handle invalid number format
                    logTextArea.append("Invalid number format: " + input + "\n");
                    isValid = false;
                } catch (IndexOutOfBoundsException e) {
                    // Handle cases where inputs are not properly structured
                    logTextArea.append("Error: Input found outside a defined layer: " + input + "\n");
                    isValid = false;
                }
            }
        }
        if (convertedInputs.isEmpty()) {
            isValid = false;
            logTextArea.append("Can't train an empty network\n");
        }

        System.out.println(isValid);
        return isValid; // Return whether all fields are valid
    }

    private static void processPanelComponents(Container container, Stack<String> inputsStack) {
        // Iterate through each component in the panel/container
        for (Component component : container.getComponents()) {
            if (component instanceof JPanel blockPanel) {
                // Process the block panel and its components recursively
                processBlockPanel(blockPanel, inputsStack);
            }
        }
    }

    private static void processBlockPanel(JPanel blockPanel, Stack<String> inputsStack) {
        // Get the block label (name) from the first component in the panel, which should be a JLabel
        Component[] blockComponents = blockPanel.getComponents();
        if (blockComponents.length > 0 && blockComponents[0] instanceof JLabel blockLabel) {
            // Push the block label onto the stack
            inputsStack.push(blockLabel.getText());

            // Recursively process each child component in the block panel
            processChildComponents(blockPanel, inputsStack);
        }
    }

    private static void processChildComponents(JPanel blockPanel, Stack<String> inputsStack) {
        // Iterate over the child components of the block panel
        for (Component child : blockPanel.getComponents()) {
            if (child instanceof JTextField textField) {
                // Push the value of the JTextField onto the stack
                if (textField.getText().contains("Layer")) {
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

    private static JPanel addFilePanel() {
        JPanel filePanel = new JPanel();
        filePanel.setLayout(new BoxLayout(filePanel, BoxLayout.Y_AXIS)); // Stack vertically

        JLabel filePathLabel = new JLabel("Folder Path:");  // Label for the folder path
        JTextField filePathTextField = new JTextField(30);  // Text field to display the selected folder path
        filePathTextField.setEditable(false);  // Make it non-editable
        filePathTextField.setText("No folder selected");  // Placeholder text

        JButton fileSelectButton = new JButton("Select Folder");
        fileSelectButton.addActionListener(_ -> {
            JFileChooser folderChooser = new JFileChooser();
            folderChooser.setDialogTitle("Select a Folder");
            folderChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);  // Allow only folder selection
            int result = folderChooser.showOpenDialog(null);
            if (result == JFileChooser.APPROVE_OPTION) {
                File selectedFolder = folderChooser.getSelectedFile();
                filePathTextField.setText(selectedFolder.getAbsolutePath());  // Display the selected folder path
                startButton.setEnabled(true);
                filepath = selectedFolder.getAbsolutePath();
            }
        });

        filePanel.add(filePathLabel);
        filePanel.add(filePathTextField);
        filePanel.add(fileSelectButton);
        return filePanel;
    }

    private static JPanel getBlockPanel(String block) {
        JPanel blockPanel = new JPanel();
        blockPanel.setLayout(new GridLayout(0, 2)); // Explicit grid layout with enough rows and 2 columns
        blockPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        blockPanel.setOpaque(true);

        // Add Block Label at the top (span two columns)
        JLabel blockLabel = new JLabel(block, SwingConstants.CENTER);
        blockPanel.add(blockLabel);  // First column
        blockPanel.add(new JLabel()); // Spacer for the second column to keep alignment

        // Parameters section
        switch (block) {
            case "Convolutional Layer" -> {
                blockPanel.add(new JLabel(" Kernel Size:"));        // First column
                blockPanel.add(new JTextField(5));                  // Second column

                blockPanel.add(new JLabel(" Stride:"));             // First column
                blockPanel.add(new JTextField(5));                  // Second column

                blockPanel.add(new JLabel(" Weight Scale:"));       // First column
                blockPanel.add(new JTextField(5));                  // Second column

                blockPanel.add(new JLabel(" Seed:"));               // First column
                blockPanel.add(new JTextField(5));                  // Second column
            }
            case "Pooling Layer" -> {
                blockPanel.add(new JLabel(" Kernel Size:"));       // First column
                blockPanel.add(new JTextField(5));            // Second column

                blockPanel.add(new JLabel(" Padding:"));          // First column
                blockPanel.add(new JTextField(5));            // Second column
            }
            case "Fully Connected Layer" -> {
                blockPanel.add(new JLabel(" Weight Init Scale:"));   // First column
                blockPanel.add(new JTextField(5));                   // Second column

                blockPanel.add(new JLabel(" Bias Init Value:"));     // First column
                blockPanel.add(new JTextField(5));                   // Second column

                blockPanel.add(new JLabel(" Seed:"));                // First column
                blockPanel.add(new JTextField(5));                   // Second column
            }
        }

        // Set TransferHandler for drag-and-drop functionality
        blockPanel.setTransferHandler(new ComponentTransferHandler());
        blockPanel.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent evt) {
                JComponent component = (JComponent) evt.getSource();
                TransferHandler handler = component.getTransferHandler();
                handler.exportAsDrag(component, evt, TransferHandler.MOVE);
            }
        });

        return blockPanel;
    }

    public static void main(String[] args) {
        CNNPlayground gui = new CNNPlayground();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Set the window to close on exit
        gui.setTitle("CNN Playground"); // Set the window title
        gui.setVisible(true); // Make the window visible
        gui.setSize(new Dimension(1300, 1000));
        gui.setLocation(200, 50); // Position the window on the screen
    }

    // Custom TransferHandler for Component Drag-and-Drop
    private static class ComponentTransferHandler extends TransferHandler {
        private static final DataFlavor COMPONENT_FLAVOR = new DataFlavor(JComponent.class, "JComponent");

        // Recursion
        private static void printPanelLayoutsRecursively(JComponent component) {
            if (component instanceof JPanel panel) {
                System.out.println("Panel: " + panel.getLayout());
                for (Component child : panel.getComponents()) {
                    printPanelLayoutsRecursively((JComponent) child); // Recurse through child components
                }
            }
        }

        @Override
        protected Transferable createTransferable(JComponent c) {
            return new ComponentTransferable(c);
        }

        @Override
        public int getSourceActions(JComponent c) {
            return MOVE;
        }

        @Override
        public boolean canImport(TransferSupport support) {
            // Check the drop target component
            Component dropTarget = support.getComponent();

            // Ensure the target is a JPanel
            if (dropTarget instanceof JPanel targetPanel) {
                Border border = targetPanel.getBorder();

                // Check if the panel has a TitledBorder and compare its title
                if (border instanceof TitledBorder) {

                    String panelTitle = ((TitledBorder) border).getTitle();
                    if (panelTitle.contains("Drag & Drop Area")) {
                        return support.isDataFlavorSupported(COMPONENT_FLAVOR);
                    } else {
                        return false;
                    }
                }
            }

            return super.canImport(support);
        }

        @Override
        public boolean importData(TransferSupport support) {
            if (!canImport(support)) {
                return false;
            }

            try {
                JComponent component = (JComponent) support.getTransferable().getTransferData(COMPONENT_FLAVOR);

                printPanelLayoutsRecursively(component);

                Container targetContainer = (Container) support.getComponent();

                // Remove from old container
                Container parent = component.getParent();
                if (parent != null) {
                    parent.remove(component);
                }

                component.addMouseListener(new java.awt.event.MouseAdapter() {
                    private boolean isHighlighted = false;
                    private Point initialClick;
                    private Point offset;

                    @Override
                    public void mouseClicked(java.awt.event.MouseEvent evt) {
                        if (SwingUtilities.isRightMouseButton(evt)) {
                            // Right-click: Delete the component
                            targetContainer.remove(component);
                            targetContainer.revalidate();
                            targetContainer.repaint();
                        } else if (SwingUtilities.isLeftMouseButton(evt)) {
                            // Left-click: Toggle highlight
                            if (isHighlighted) {
                                component.setBackground(Color.WHITE);
                                isHighlighted = false;
                            } else {
                                component.setBackground(Color.YELLOW); // Highlight
                                isHighlighted = true;
                            }
                        }
                    }

                    @Override
                    public void mousePressed(java.awt.event.MouseEvent evt) {
                        if (isHighlighted && SwingUtilities.isLeftMouseButton(evt)) {
                            // Store the initial click point and component's location for dragging
                            initialClick = evt.getPoint();
                            offset = component.getLocation();
                        }
                    }

                    @Override
                    public void mouseDragged(java.awt.event.MouseEvent evt) {
                        if (isHighlighted && initialClick != null && SwingUtilities.isLeftMouseButton(evt)) {
                            // Calculate and set the new position during dragging
                            int dx = evt.getX() - initialClick.x;
                            int dy = evt.getY() - initialClick.y;
                            Point newLocation = new Point(offset.x + dx, offset.y + dy);
                            component.setLocation(newLocation);

                            targetContainer.repaint(); // Update the container to reflect the change
                        }
                    }

                    @Override
                    public void mouseReleased(java.awt.event.MouseEvent evt) {
                        if (isHighlighted && SwingUtilities.isLeftMouseButton(evt)) {
                            // Handle drop logic after dragging
                            Point currentPoint = evt.getPoint();
                            Point targetPoint = SwingUtilities.convertPoint(component, currentPoint, targetContainer);

                            boolean placedInTarget = false;

                            // Check if the drop position is valid within the target container
                            for (Component c : targetContainer.getComponents()) {
                                if (c != component && c.getBounds().contains(targetPoint)) {
                                    // Reorder component to the drop position
                                    targetContainer.remove(component);
                                    targetContainer.add(component, targetContainer.getComponentZOrder(c));
                                    targetContainer.revalidate();
                                    targetContainer.repaint();
                                    placedInTarget = true;
                                    break;
                                }
                            }

                            if (!placedInTarget) {
                                // If no valid position, reset to the original location
                                component.setLocation(offset);
                            }
                        }
                    }
                });

                // Initially add to the new container (leftPanel)
                targetContainer.add(component);
                targetContainer.revalidate();
                targetContainer.repaint();
                return true;
            } catch (Exception e) {
                System.out.println(e.getMessage());
                return false;
            }
        }

        // Transferable Implementation for Components
        private record ComponentTransferable(JComponent component) implements Transferable, Serializable {

            @Override
            public DataFlavor[] getTransferDataFlavors() {
                return new DataFlavor[]{COMPONENT_FLAVOR};
            }

            @Override
            public boolean isDataFlavorSupported(DataFlavor flavor) {
                return COMPONENT_FLAVOR.equals(flavor);
            }

            @Override
            public Object getTransferData(DataFlavor flavor) {
                return component;
            }
        }
    }

    public static class layerEntry {
        private final String layer;
        private ArrayList<Double> list;

        public layerEntry(String layer, ArrayList<Double> list) {
            this.list = list;
            this.layer = layer;
        }

        public String getLayer() {
            return layer;
        }

        public ArrayList<Double> getList() {
            return list;
        }

        public void setList(ArrayList<Double> list) {
            this.list = list;
        }

        public String toString() {
            return "LayerEntry{" +
                    "layer='" + layer + '\'' +
                    ", list=" + Arrays.toString(list.toArray()) +
                    '}';
        }
    }
}

/*
ToDo:
    - add training process

    - change seed
    - remove box logic
 */