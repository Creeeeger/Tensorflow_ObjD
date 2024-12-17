package org.object_d;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.io.File;
import java.io.Serializable;

public class CNNPlayground extends JFrame {
    static JButton startButton;
    static JTextArea logTextArea;
    static String filepath;

    public CNNPlayground() {
        setLayout(new GridLayout(1, 2));
        setTitle("CNN Playground");

        JPanel leftPanel = new JPanel();
        leftPanel.setBorder(BorderFactory.createTitledBorder("Drag & Drop Area - right click for deleting - left click for highlighting and moving"));
        leftPanel.setLayout(new GridLayout(8, 1, 0, 5)); // Vertical gap of 10 pixels
        leftPanel.setBackground(Color.LIGHT_GRAY);
        leftPanel.setEnabled(true);

        // Right Panel: Actions and Block Area
        JPanel rightPanel = new JPanel(new GridLayout(2, 1));
        rightPanel.setBorder(BorderFactory.createTitledBorder("Actions Area"));

        // Block Area
        JPanel blockArea = new JPanel(new GridLayout(3, 2));
        blockArea.setBorder(BorderFactory.createTitledBorder("Block Area"));
        JLabel DaDInfo = new JLabel("<html>Drag and drop the blocks below to form a CNN.<br>Set parameters and train the model.</html>");
        DaDInfo.setHorizontalAlignment(SwingConstants.CENTER);  // Center the text
        blockArea.add(DaDInfo);

        // Add draggable blocks
        String[] blockNames = {"Conv Layer", "Pooling Layer", "Dense Layer", "Activation Layer"};
        for (String block : blockNames) {
            JPanel blockPanel = getBlockPanel(block);
            blockArea.add(blockPanel);
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
            logTextArea.append("Start button clicked!\n");
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
            case "Conv Layer" -> {
                blockPanel.add(new JLabel(" Filters:"));         // First column
                blockPanel.add(new JTextField(5));            // Second column

                blockPanel.add(new JLabel(" Kernel Size:"));     // First column
                blockPanel.add(new JTextField(5));            // Second column

                blockPanel.add(new JLabel(" Stride:"));          // First column
                blockPanel.add(new JTextField(5));            // Second column
            }
            case "Pooling Layer" -> {
                blockPanel.add(new JLabel(" Pool Size:"));       // First column
                blockPanel.add(new JTextField(5));            // Second column

                blockPanel.add(new JLabel(" Stride:"));          // First column
                blockPanel.add(new JTextField(5));            // Second column
            }
            case "Dense Layer" -> {
                blockPanel.add(new JLabel(" Units:"));           // First column
                blockPanel.add(new JTextField(5));            // Second column
            }
            case "Activation Layer" -> {
                blockPanel.add(new JLabel(" Activation:"));    // First column
                blockPanel.add(new JLabel("ReLU"));            // Second column
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
            return support.isDataFlavorSupported(COMPONENT_FLAVOR);
        }

        @Override
        public boolean importData(TransferSupport support) {
            if (!canImport(support)) {
                return false;
            }

            try {
                JComponent component = (JComponent) support.getTransferable().getTransferData(COMPONENT_FLAVOR);
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
}

/*
ToDo:
    - remove duplication error on right side
    - do input check and get input
    - do layer check
    - add layer extraction
    - link process later
 */