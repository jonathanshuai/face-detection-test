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

#Define some filenames and constants
face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
input_dir = "./images"
output_dir = "./aligned"
input_dir_len = len(input_dir) + 1

openface_path = "../building/openface/batch-represent/main.lua"
openface_outdir =  "-outDir"
openface_outdir_arg = "./reps" 
openface_data_dir = "-data"
openface_data_dir_arg = "./aligned"

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

    #If we couldn't find an image, give a warning and continue
    if image_cropped is None:
      warnings.warn('No face found for {}'.format(image_path), UserWarning)
      continue

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

subprocess.call([openface_path, openface_outdir, openface_outdir_arg, 
                openface_data_dir, openface_data_dir_arg])


