import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt

import time 

import cv2

#Define some filenames and constants
csv_file = "data.csv"
lbpcascade = "lbpcascade_frontalface_improved.xml"
eyecascade = "haarcascade_eye.xml"
jisoo_id = "js"
seolhyun_id = "sh"


#Read in the data to a pandas dataframe
df = pd.read_csv(csv_file)

#Get only Jisoo and Seolhyun images
#df = df.loc[(df['id']==jisoo_id) | (df['id'] == seolhyun_id)]
df = df.loc[(df['id']==jisoo_id)]
df = df.sort_values('path')


#Iterate through each picture
for i, r in df.iterrows():
  image_file = r['path']

  #read image in grayscale (flag 0)
  img_c = cv2.imread(image_file)
  img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
  


  #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
  #eyes = eye_cascade.detectMultiScale(img_g, scaleFactor=1.05)

  #for (x, y, w, h) in eyes:
  #  cv2.rectangle(img_c, (x, y), (x+w, y+h), (255, 0, 0), 2)



  face_cascade = cv2.CascadeClassifier(lbpcascade)
  faces = face_cascade.detectMultiScale(
    img_g,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
  )

  for (x, y, w, h) in faces:
    cv2.rectangle(img_c, (x, y), (x+w, y+h), (0, 255, 0), 2)


  cv2.imshow('image', img_c)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  





