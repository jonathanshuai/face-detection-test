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
import openface	

from FacePreprocessor import FacePreprocessor

#Define some filenames and constants
csv_file = "data.csv"
face_landmark_dat = "./sets/shape_predictor_68_face_landmarks.dat"

jisoo_id = "js"
seolhyun_id = "sh"
iu_id = "iu"
mf_id = "mf"

#Read in the data to a pandas dataframe
df = pd.read_csv(csv_file)

#Get only Jisoo and Seolhyun images
#df = df.loc[(df['id'] == jisoo_id) | (df['id'] == seolhyun_id)]
#df = df.loc[(df['id'] == jisoo_id)].head(4)
#df = df.loc[(df['id'] == seolhyun_id)]
#df = df.loc[(df['path'] == './images/jisoo/jisoo13.jpg')]
df = df.loc[(df['id'] == jisoo_id) | (df['id'] == iu_id)]
#df = df.loc[(df['id'] == mf_id) | (df['id'] == iu_id)]

parameters = {'landmark_dat': face_landmark_dat, 
'left_eye_pos': (0.34, 0.34), 'width': 300, 'height': 300,
'clahe_clip_limit': 2.0, 'clahe_tile_grid_size': (8, 8)}

fp = FacePreprocessor(**parameters)

processed_faces = []
face_labels = []
for index, row in df.iterrows():
  #Read in the image
  image_file = row['path']
  image_color = cv2.imread(image_file)
  
  #Crop and align the face from the image
  cropped_images = fp.crop_and_align(image_color)

  for image_cropped in cropped_images: #there should only be one!!
    #Apply contrast limiting and smoothing
    image_cropped = fp.apply_clahe(image_cropped)
    fp.build_gamma_table(1.7)
    image_cropped = fp.apply_gamma_correction(image_cropped)
    image_cropped = fp.apply_smoothing(image_cropped, (3, 3), 0.788, 0.788)
    #image_cropped = fp.apply_canny(image_cropped, 30, 210)

    #image_cropped = fp.apply_dog(image_cropped, (3, 3), 0.788, 0.788,
    #                                            (5, 5), 1.028, 1.028)

    #Add the processed face and label to our list
    processed_faces.append(image_cropped.flatten())
    face_labels.append(row['label'])

    #cv2.imshow(row['path'], image_color)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow("image_smoothed", image_cropped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



