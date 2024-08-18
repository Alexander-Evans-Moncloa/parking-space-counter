import os
from skimage.io import imread               #Allows us to read files from the computer
from skimage.transform import resize        #Allows us to resize images. All of this without OpenCV (cv2)
import numpy as np
from sklearn.model_selection import train_test_split       #This allows us to train and test the data
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#Data preparation

inputdir = '/home/alexander/Documents/CV_ImageClassification/Data'      #Where the raw data should be
categories = ['empty', 'not_empty']                                     #Classifications of data. In this case only 2.

data = []       #Object for data to later be put into it
labels = []     #Object for label

for category_idx, category in enumerate(categories):            #Loops as many times as there are categories (only twice in this case). Has 2 increasing numbers.
    for file in os.listdir(os.path.join(inputdir, category)):   #Loops as many times as there are items in the input directory. os.listdir() gets lists everything in a folder
        img_path = os.path.join(inputdir, category, file)       #Determines the path of the file itself, creates it by adding "category" number then "file" number to the folder
        img = imread(img_path)                                  #Reads image from image path, saves as variable
        img = resize(img, (15,15))                              #Resizes all images
        data.append(img.flatten())                              #Adds an array version of the image into the object "data". Flatted makes matrix into an array (1 long row)
        labels.append(category_idx)                             #Adds the category index to the labels object

data = np.asarray(data)             #Casts each list into a numpy array
labels = np.asarray(labels)         #Same as above

#Train / test split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)       #Split size of 0.2 means it is 80% training set, 20% test set. Shuffling is good practise (avoids biases when creating data sets). Stratifying makes sure all the different labels are in the same proportion as the og dataset

#Train classifier

classifier = SVC()      #Creates new instance of SVC, calls instance classifier (the classifier we are going to use)

parameters = [{'gamma': [0.01, 0.001, 0.0001],'C': [1, 10, 100, 1000]}]     #Creates object called "parameters", list containing only a dictionary. With 2 keys (gamma and C) with a list of values.

grid_search = GridSearchCV(classifier, parameters)      #Uses GridSearchCV to train MANY different image classifiers, one for each combo of gamma and C (in this case, 3*4 = 12)

grid_search.fit(x_train, y_train)

#Test performance

best_estimator = grid_search.best_estimator_            #Selects one of the 12 classifiers. It gets the BEST of all the 12 trained image classifiers and calls it best_estimator (using grid search)

y_prediction = best_estimator.predict(x_test)           #Applies the best estimators predictions to the x testing block (20% of remaining data) and saves the y predictions

score = accuracy_score(y_prediction, y_test)            #Creates a score by using the functon "accuracy_score" from skikit learn -> metrics that compares the real y test data with predicted data

print('{}% of samples were correctly classified'.format(str(score * 100)))      #Outputs score value (it's a value between 0 and 1) with some nicer formatting

pickle.dump(best_estimator, open('./ParkingLotDetectorAndCounter/MyModel/model.p','wb'))      #Saves the object we want to save (the best estimator) using pickle.dump, specifies the file where it will dump the model (file is called wb, model is model.p)