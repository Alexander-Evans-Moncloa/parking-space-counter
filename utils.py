import pickle
from skimage.transform import resize
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("./ParkingLotDetectorAndCounter/MyModel/model.p", "rb"))      #Loads our machine learnt model from pickle (I think). Make sure this is actually the same thing!!!

#Change to ./ParkingLotDetectorAndCounter/MyModel/model.p to use my home grown one (working!), ./ParkingLotDetectorAndCounter/model/model.p for Felipe's one

def empty_or_not(spot_bgr):                     #Creates function called "empty or not" and imports the image entered and names it "spot bgr"

    flat_data = []                              #Creates variable/object for the flat data to later be put into

    img_resized = resize(spot_bgr, (15, 15, 3)) #Resizes the "spot bgr" image and renames it "img resized"

    flat_data.append(img_resized.flatten())     #Adds a flattened img_resized to flat data

    flat_data = np.array(flat_data)             #Makes the flat data into a numpy array to make sure its NOT A MATRIX but 1 row array :)

    y_output = MODEL.predict(flat_data)         #It applies the imported model.p (MODEL) and recalls the .predict function, applies it to the flat data and saves it as y_output

    if y_output == 0:                           #If else statement to dump the y_output data as either EMPTY or NOT_EMPTY depending on the value
        return EMPTY
    else:
        return NOT_EMPTY                        #Example: y_output value is 0.9 then it knows something is inside, therefore NOT_EMPTY
    

def get_parking_spots_bboxes(connected_components):                         #Creates function to get parking spots' bounding boxes with input of the connected components

    (totalLabels, label_ids, values, centroid) = connected_components       #Uses the connected components object and declares 4 variables from the object

    slots = []                                                              #Declares slot matrix (not a numpy array though)

    coef = 1                                                                #Declares coefficient variable and sets it as 1

    for i in range(1, totalLabels):

        #This extracts the coordinate points for the number of labels there are
        x1 = int(values[i, cv2.CC_STAT_LEFT]*coef)
        y1 = int(values[i, cv2.CC_STAT_TOP]*coef)                           #Gets the integer of the values in connected_components at point i to cv2.CC_STAT_TOP multiplied by a coefficient
        w = int(values[i, cv2.CC_STAT_WIDTH]*coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT]*coef)

        slots.append([x1, y1, w, h])                                        #Adds the recently extracted coordinate points to the slots array

    return slots                                                            #Returns array to main code
