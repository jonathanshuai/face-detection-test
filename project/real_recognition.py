import os
import time 
import warnings
import subprocess

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import cv2
import dlib
import imutils
from imutils import face_utils

from FacePreprocessor import FacePreprocessor

#Define some filenames, directories, and constants
face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
input_dir = "./images"
output_dir = "./aligned"
input_dir_len = len(input_dir) + 1

#The data and output directories for openface embedding script
openface_path = "../building/openface/batch-represent/main.lua"
openface_outdir =  "-outDir"
openface_outdir_arg = "./reps" 
openface_data_dir = "-data"
openface_data_dir_arg = "./aligned2"

parameters = {'landmark_dat': face_landmark_dat, 
'left_eye_pos': (0.37, 0.37), 'lip_pos': 0.82, 'width': 96, 'height': 96}
fp = FacePreprocessor(**parameters)

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

for subdir, dirs, files in os.walk(input_dir):
  for file in files:
    #Read in the image
    input_image_file = os.path.join(subdir, file)
    image_color = cv2.imread(input_image_file)

    print("Cropping and aligning image {} ...".format(input_image_file))

    #Crop and align the face from the image (returns just ONE face (the largest))
    image_cropped = fp.crop_and_align(image_color, get_one=True)

    #If we couldn't find an image, give a warning and continue
    if image_cropped is None:
      warnings.warn('No face found for {}'.format(image_path), UserWarning)
      continue
    else:
      #Since crop_and_align returns an array, get the first one
      image_cropped = image_cropped[0]

    #Get/create the output directory and filename
    output_image_file = os.path.join(output_dir, input_image_file[input_dir_len:])
    output_image_dir = os.path.dirname(output_image_file)
    if not os.path.exists(output_image_dir):
     	os.makedirs(output_image_dir)

    #Show the cropped image
    #cv2.imshow("image_smoothed", image_cropped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #Write the image to the designated diretory
    cv2.imwrite(output_image_file, image_cropped)

print("Finished cropping and aligning.")
print("Calling openface's lua script from {} ...".format(openface_path))

#Call openface lua script
subprocess.call([openface_path, openface_outdir, openface_outdir_arg, 
                openface_data_dir, openface_data_dir_arg])

#print("openface lua script finished.")

#Read in the representation and the labels from the csv generated from openface
reps = pd.read_csv(os.path.join(openface_outdir_arg, 'reps.csv'), header=None)
labels = pd.read_csv(os.path.join(openface_outdir_arg, 'labels.csv'), header=None)

#Create the X and y of our dataset
X = np.array(reps)
y = labels[0] - 1

#Encode our labels with LabelEncoder 
le = LabelEncoder()
#le.fit(y)
le.fit(['Morgan Freeman', 'Jisoo', 'Jennie', 'Lisa', 'Rose', 'IU', 'Seolhyun'])
n_classes = len(le.classes_)
#y = le.transform(y)

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

parameters =  {'C': 1000, 'kernel': 'linear'}
clf = SVC(**parameters)
clf.fit(X_train.astype('float'), y_train.astype('float'))



training_predictions = clf.predict(X_train)
training_score = accuracy_score(training_predictions, y_train)
print(training_score)

testing_predictions = clf.predict(X_test)
testing_score = accuracy_score(testing_predictions, y_test)
print(testing_score)


testing_predictions = le.inverse_transform(testing_predictions.astype('int'))
y_test = le.inverse_transform(y_test.astype('int'))

for (a, b) in zip(testing_predictions, y_test):
  print("Guessed: {} | Answer: {}".format(a,b))

