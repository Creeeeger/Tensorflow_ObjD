package org.tensorAction;

import org.tensorflow.SavedModelBundle;

public class detector {
    public String[] classify(String imagePath, SavedModelBundle ModelBundle) {

        //Base logic for returning image path and labels
        String path = "";
        String result = "";
        String[] returnArray = new String[2];
        returnArray[0] = path;
        returnArray[1] = result;

        //Add whole image detection logic (detect_ev image recogniser event)!!!

        return returnArray;
    }
}
//Add whole image detection logic (detect_ev image recogniser event)!!!
