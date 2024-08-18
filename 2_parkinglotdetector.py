import cv2                                      #Imports OpenCV 2
from utils import get_parking_spots_bboxes      #Imports function from the utils.py code
from utils import empty_or_not                  #Same as above
import numpy as np                              #Imports numpy

def calc_diff(im1, im2):                        #Function calculates the difference between image 1 and image 2
    return np.abs(np.mean(im1) - np.mean(im2))  #Returns the average number of all pixels in image 1 - the average of image 2. Rough estimation. Pos num only.


mask = './ParkingLotDetectorAndCounter/parking-space-counter-master/mask_1920_1080.png'  #Finds mask from files

video_path = './ParkingLotDetectorAndCounter/data/parking_1920_1080_loop.mp4'            #Finds video path from files

mask = cv2.imread(mask, 0)              #Loads mask from image, converts to black and white (it already is)

cap = cv2.VideoCapture(video_path)      #Captures video

ret = True                              #Variable indicating video isn't over yet

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)     # This function finds all the connected components in the binary mask image.
# 'mask' is the input image, where it looks for white regions (assumed to be parking spaces).
# '4' specifies the connectivity type (4-connected neighbors, meaning the pixels connect horizontally or vertically).
# 'cv2.CV_32S' is the data type used for the output labels (32-bit signed integer).
# The function returns several outputs, including labels and statistics for each connected component, 
# which are used to identify and measure distinct parking spots in the mask.

#But what is "connectedComponents" you say? Here is a description of the maths: A group of connected nodes is called a "connected component".
#In our example, all connected white pixels that form a box are "connected components", therefore the mask allows us to determine where the parking spots are

spots = get_parking_spots_bboxes(connected_components)      #Gets the parking spots by going through all the components it has found and gets the bounding box from each one

print(spots[0])                         #Prints the first spot

spots_status = [None for j in spots]    #Initializes an array 'spots_status' with the same length as the 'spots' list.
# Each element of 'spots_status' is initially set to None and will be updated later to indicate whether each parking spot is occupied or not.

diffs = [None for j in spots]           #Saves value to measure if something is happening in the parking lot or not.

previous_frame = None                   #Every time we iterate, we save the previous frame here


step = 30                               #Declares how many frames will pass before it reclassifies the parking lots

frame_nmr = 0                           #Initialises frame number variable

while ret:                              #While the capture is still running ret is True
    ret, frame = cap.read()             #Reads captured video

    #This part computes the difference between current frame and previous frame, then saves it to diffs (differences)
    if frame_nmr % step == 0 and previous_frame is not None:   #Remainder of 0 in the division between frame number and steps
        for spot_indx, spot in enumerate(spots):            #spot_indx is the index of the current spot in the spots list, and spot contains the bounding box data for that spot
            x1, y1, w, h = spot                             #Creates crop again
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]      #Comments below

            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :]) #Saves, into diffs (with the spot index), the calc diff between a crop of frame and previous frame

        #print([diffs[j] for j in np.argsort(diffs)][::-1])  #Prints diffs array, ordered by size, biggest first

    #This should only be done if the difference is large
    if frame_nmr % step == 0:                               #If the remainder of a division of the frame number by 30 is 0, then its 30, 60, 90 etc and it runs

        if previous_frame is None:                          #Only if we are in the first frame (because if prev frame is none it breaks code later down)
            arr_ = range(len(spots))                        #Make this new array equal to the range of the length of spots
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4][::-1]
            #Moves through all spots ONLY IN SPOTS GIVEN BY A DIFFERENCE VALUE OVER 0.4. As j increases through diffs (the index) it gets divided by the maximum value for the frame. If its above 0.4, execute.
        
        for spot_indx in arr_:                              #Goes through array, which will definitely have a value in it

            spot = spots[spot_indx]                         #As we didn't define spot in the for loop, we now make it equal to the value of object spots at index [spot_indx]

            x1, y1, w, h = spot                             #Takes all 4 values from the spots object

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]      #Crops the frame in the required dimensions (from y1 to that point + height, and same for x)

            spot_status = empty_or_not(spot_crop)           #Determines if the spot being cropped is empty or not using model.p

            spots_status[spot_indx] = spot_status           #Adds current spot_status to the INDEX of the spot_status list, which keeps track of what parking spot is empty or not

    #This bit needs to be done before the drawing because after the drawing, the frame gets changed
    if frame_nmr % step == 0:                           #If we are updating the spot status
        previous_frame = frame.copy()                   #Saves frame into the previous frame (by making a copy)


    for spot_indx, spot in enumerate(spots):            #This is done continuously because if not it would just show boxes once every 30th frame

        spot_status = spots_status[spot_indx]           #Gets status of current parking spot (in the index of spot status) and declares that as the variable (unpacking it)

        x1, y1, w, h = spots[spot_indx]                 #Unpacks bounding box data for current parking lot spot (spots[spot_indx])

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)    #Draws red rectangle in the spot    
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)    #Uses the variables to draw a red rectangle around the frame

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #The above takes the sum of the spots_status and divides it by the length of spots_status. This is because "spots_status" is a boolea (true/false) so every true counts as 1 so they all add to 124 or whatever.

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)             #Prevents feed from going off the screen
    cv2.imshow('frame', frame)                              #Shows current frame being read
    if cv2.waitKey(25) & 0xFF == ord('p'):                  #Waits 25 milliseconds to show next frame, will shut down if 'p' is pressed
        break

    frame_nmr += 1                                      #Adds 1 to frame number




cap.release()               #Releases capture
cv2.destroyAllWindows()     #Closes windows