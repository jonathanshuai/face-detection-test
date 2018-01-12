import os
import time 
import warnings

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

#Define some filenames and constants
face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
input_dir = "./images"
output_dir = "./aligned"

jisoo_id = "js"
seolhyun_id = "sh"
iu_id = "iu"
mf_id = "mf"


#Get only Jisoo and Seolhyun images
#df = df.loc[(df['id'] == jisoo_id) | (df['id'] == seolhyun_id)]
#df = df.loc[(df['id'] == jisoo_id)].head(4)
#df = df.loc[(df['id'] == seolhyun_id)]
#df = df.loc[(df['path'] == './images/jisoo/jisoo13.jpg')]
#df = df.loc[(df['path'] == './iu/iu1.jpg')]
#df = df.loc[(df['id'] == jisoo_id) | (df['id'] == iu_id)]
#df = df.loc[(df['id'] == iu_id)]
#df = df.loc[(df['id'] == mf_id) | (df['id'] == iu_id)]

print(os.path.join(subdir, file))
        print(subdir)
        print(file)
parameters = {'landmark_dat': face_landmark_dat, 
'left_eye_pos': (0.37, 0.37), 'lip_pos': 0.82, 'width': 200, 'height': 200}

fp = FacePreprocessor(**parameters)

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

for subdir, dirs, files in os.walk(input_dir):
  for file in files:
	  #Read in the image
	  input_image_file = os.path.join(subdir, file)
	  image_color = cv2.imread(input_image_file)
	  
	  #Crop and align the face from the image (returns just ONE face (the largest))
	  image_cropped = fp.crop_and_align(image_color)

	  if image_cropped is None:
	    warnings.warn('No face found for {}'.format(image_path), UserWarning)


	  #Add the processed face and label to our list
	  #processed_faces.append(image_cropped.flatten())
	  #face_labels.append(row['label'])
	  output_image_file = os.path.join(output_dir, input_image_file[len(input_dir):])
	  output_image_dir = os.path.dirname(output_image_file)
	  if not os.path.exists(output_image_dir):
	  	os.makedirs(output_image_dir)


	  #cv2.imshow("image_smoothed", image_cropped)
	  #cv2.waitKey(0)
	  #cv2.destroyAllWindows()

	  cv2.imwrite(output_image_file, image_cropped)


