import time 
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt

import cv2
import imutils
from imutils import face_utils
import dlib

from FacePreprocessor import FacePreprocessor

#Define some filenames and constants
csv_file = "data.csv"
face_landmark_dat = "./sets/shape_predictor_68_face_landmarks.dat"

left_eye_position = (0.3, 0.3)
crop_size = 300

jisoo_id = "js"
seolhyun_id = "sh"
songhyekyo_id = "shk"
hanhyojoo_id = "hhj"

#Read in the data to a pandas dataframe
df = pd.read_csv(csv_file)

#Get only Jisoo and Seolhyun images
#df = df.loc[(df['id'] == jisoo_id) | (df['id'] == seolhyun_id)]
#df = df.loc[(df['id'] == jisoo_id) | (df['id'] == songhyekyo_id)]
#df = df.loc[(df['id'] == hanhyojoo_id) | (df['id'] == songhyekyo_id)]
df = df.loc[(df['id'] == jisoo_id)]
#df = df.loc[(df['id'] == seolhyun_id)]
#df = df.sort_values('path')

#df = df.loc[(df['path'] == './images/jisoo/jisoo13.jpg')]

fp = FacePreprocessor(face_landmark_dat)


for index, row in df.iterrows():
  image_file = row['path']
  image_color = cv2.imread(image_file)
  cropped_images = fp.crop_and_align(image_color)
  for image_cropped in cropped_images:
    cv2.imshow("image_cropped", image_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_corrected = fp.apply_clahe(image_cropped)
    cv2.imshow("image_cropped", image_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








