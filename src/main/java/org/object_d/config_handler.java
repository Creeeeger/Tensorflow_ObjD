package org.object_d;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;

public class config_handler {
    public static String[][] load_config() {
        try {
            File inputFile = new File("config.xml");
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            org.w3c.dom.Document doc = dBuilder.parse(inputFile);
            doc.getDocumentElement().normalize();

            // Get all child nodes of the <config> element
            NodeList nodeList = doc.getDocumentElement().getChildNodes();

            // Count the number of element nodes
            int numEntries = 0;
            for (int i = 0; i < nodeList.getLength(); i++) {
                if (nodeList.item(i).getNodeType() == Node.ELEMENT_NODE) {
                    numEntries++;
                }
            }

            // Initialize the array based on the number of entries
            String[][] values = new String[numEntries][2]; // Each entry has two values: name and content

            // Load data from XML into the array
            int entryIndex = 0;
            for (int i = 0; i < nodeList.getLength(); i++) {
                Node node = nodeList.item(i);
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    values[entryIndex][0] = node.getNodeName(); // Store the name of the element
                    values[entryIndex][1] = node.getTextContent(); // Store the text content of the element
                    entryIndex++;
                }
            }

            return values;

        } catch (Exception e) {
            e.printStackTrace();
            create_config();
            return null; // Return null if an error occurs
        }
    }

    public static void save_config(String[][] values) {
        try {
            DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

            // Create a new Document
            Document document = documentBuilder.newDocument();

            // Create the root element
            Element rootElement = document.createElement("config");
            document.appendChild(rootElement);

            // Create child elements and set their values
            for (String[] entry : values) {
                String name = entry[0];
                String content = entry[1];

                Element element = document.createElement(name);
                element.appendChild(document.createTextNode(content));
                rootElement.appendChild(element);
            }

            // Transform the Document into XML format
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");

            DOMSource domSource = new DOMSource(document);
            StreamResult streamResult = new StreamResult(new File("config.xml"));
            transformer.transform(domSource, streamResult);

            System.out.println("Config file saved successfully!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void create_config() {
        //Add required config values!!!

        try {
            DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = builderFactory.newDocumentBuilder();
            Document doc = builder.newDocument();
            Element root = doc.createElement("config");
            doc.appendChild(root);

            Element last_path = doc.createElement("img_path");
            last_path.appendChild(doc.createTextNode("/"));
            root.appendChild(last_path);

            Element database_path = doc.createElement("ts_path");
            database_path.appendChild(doc.createTextNode("/"));
            root.appendChild(database_path);

            Element setting1 = doc.createElement("setting1");
            setting1.appendChild(doc.createTextNode("true"));
            root.appendChild(setting1);

            Element setting2 = doc.createElement("setting2");
            setting2.appendChild(doc.createTextNode("true"));
            root.appendChild(setting2);

            Element setting3 = doc.createElement("setting3");
            setting3.appendChild(doc.createTextNode("true"));
            root.appendChild(setting3);

            Element setting4 = doc.createElement("setting4");
            setting4.appendChild(doc.createTextNode("true"));
            root.appendChild(setting4);

            //Set new keys on demand here
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
            DOMSource domSource = new DOMSource(doc);
            StreamResult result = new StreamResult(new File("config.xml"));
            transformer.transform(domSource, result);

            System.out.println("Config file created successfully!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
//Add required config values