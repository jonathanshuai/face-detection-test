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


import cv2
import dlib
import imutils
from imutils import face_utils

from FacePreprocessor import FacePreprocessor

#Define some filenames and constants
csv_file = "data.csv"
face_landmark_dat = "../models/shape_predictor_68_face_landmarks.dat"

#Read in the data to a pandas dataframe
df = pd.read_csv(csv_file)

parameters = {'landmark_dat': face_landmark_dat, 
'left_eye_pos': (0.3, 0.3), 'width': 300, 'height': 300}

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
    image_cropped = fp.apply_gamma_correction(image_cropped)
    image_cropped = fp.apply_smoothing(image_cropped, (3, 3), 0.788, 0.788)
    #image_cropped = fp.apply_canny(image_cropped, 30, 210)
    #fp.build_gamma_table(1.5)

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

#Facial Recoginition 
#Create our X and y dataset
n_components = 20
X = np.array(processed_faces).astype("int")
y = face_labels

#Count the number of classes and encode our labels
le = LabelEncoder()
le.fit(y)
n_classes = len(le.classes_)
y = le.transform(y)

#Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Get weights of eigenfaces 
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#Get the average weights for each face
face_weights = [[] for _ in range(n_classes)]
for (weight, label) in zip(X_train_pca, y_train):
  face_weights[label].append(weight)
face_weights = [np.array(x).mean(axis=0) for x in face_weights]



#Predict based on the distances
predictions = []
for w in X_test_pca:
  distances = [np.linalg.norm(a - w) for a in face_weights]
  print(distances)
  predictions.append(np.argmin(distances))

score = accuracy_score(predictions, y_test)
print(score)


for weight in face_weights:
  face = pca.inverse_transform(weight).reshape((300, 300)).astype("uint8")
  cv2.imshow("avg", face)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#for face in X_train_pca:
#  face = pca.inverse_transform(face).reshape((300, 300)).astype("uint8")
#  cv2.imshow("face", face)
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()  

'''
print("Training...")

parameters = { 'hidden_layer_sizes':(1024,),
               'verbose':True, 'early_stopping':True
              }
clf = MLPClassifier(**parameters).fit(X_train_pca, y_train)

#parameters = {'C':1e20, 'kernel': 'rbf', 'probability': True}
#clf = SVC(**parameters)

#clf.fit(X_train_pca, y_train)

print("Done training! Predicting...")

training_predictions = clf.predict(X_train_pca)
training_score = accuracy_score(training_predictions, y_train)
print("Training Score: {}".format(training_score))


predictions = clf.predict(X_test_pca)
probabilities = clf.predict_proba(X_test_pca)
score = accuracy_score(predictions, y_test)
print("Testing Score: {}".format(score))

print(list(predictions))
print(y_test)

for face in X_test:
  face = face.reshape(300, 300).astype("uint8")
  cv2.imshow("face", face)
  cv2.waitKey(0)
  cv2.destroyAllWindows()  



#mean_face = pca.mean_.reshape((300, 300)).astype("uint8")
#cv2.imshow("normie face", mean_face)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

f1 = []
f2 = []
for (a, b) in zip(X, y):
  if not b:
    f1.append(a)
  else:
    f2.append(a)


f1 = np.array(f1)
f2 = np.array(f2)

f1 = f1.mean(axis=0)
f2 = f2.mean(axis=0)

f1 = f1.reshape((300, 300)).astype("uint8")
cv2.imshow("0 face", f1)
cv2.waitKey(0)
cv2.destroyAllWindows()
f2 = f2.reshape((300, 300)).astype("uint8")
cv2.imshow("1 face", f2)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


