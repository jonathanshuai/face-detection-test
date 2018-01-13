import os
import shutil
import subprocess
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import cv2
import dlib
import imutils
from imutils import face_utils

import facepreprocessor
from facepreprocessor import FacePreprocessor



class FrameLabeler:
  def __init__(self, face_landmark_dat, input_dir, output_dir,
              openface_path, openface_outdir, openface_model):
    self.face_landmark_dat = face_landmark_dat
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.output_dir_end = len(output_dir) + 1
    self.input_dir_end = len(input_dir) + 1
    self.openface_path = openface_path
    self.openface_outdir_arg = "-outDir"
    self.openface_outdir = openface_outdir
    self.openface_data_dir_arg = "-data" 
    self.openface_data_dir = output_dir
    self.openface_model_arg = "-model"
    self.openface_model = openface_model
    self.fp_parameters = {'landmark_dat': face_landmark_dat, 'width': 96, 
                      'height': 96, 'alignment': facepreprocessor.NOSE}
    self.svm_parameters = {'C': [1, 10, 1e2, 1e3, 1e4, 1e5], 
                    'kernel': ['linear', 'rbf'], 
                    'gamma': [1e-2, 1e-3, 1e-4],
                    'probability': [True]}
    self.fp = FacePreprocessor(**self.fp_parameters)

  def train(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    for subdir, dirs, files in os.walk(self.input_dir):
      for file in files:
        #Read in the image
        input_image_file = os.path.join(subdir, file)
        image_color = cv2.imread(input_image_file)

        print("Cropping and aligning image {} ...".format(input_image_file))

        #Crop and align the face from the image (returns just the largest)
        box_and_image = self.fp.crop_and_align(image_color, get_one=True)

        #If we couldn't find a face in the image, give a warning and continue
        if box_and_image is None:
          warnings.warn('No face found for {}'.format(input_image_file),
                                                             UserWarning)
          continue
        else:
          (_, image_cropped) = box_and_image
          #Since crop_and_align returns an array, get the first one
          image_cropped = image_cropped[0]

        #Get and create the output directory and filename
        output_image_file = os.path.join(self.output_dir, 
                              input_image_file[self.input_dir_end:])
        output_image_dir = os.path.dirname(output_image_file)
        if not os.path.exists(output_image_dir):
         	os.makedirs(output_image_dir)

        #Write the image to the designated diretory
        cv2.imwrite(output_image_file, image_cropped)

    print("Finished cropping and aligning.")
    print("Calling openface's lua script from {} ...".format(self.openface_path))

    #Call openface lua script
    subprocess.call([self.openface_path, self.openface_outdir_arg, 
                    self.openface_outdir, self.openface_data_dir_arg, 
                    self.openface_data_dir, self.openface_model_arg, 
                    self.openface_model])

    print("openface lua script finished.")

    #Read in the reps and the labels from the csv generated from openface
    reps = pd.read_csv(os.path.join(self.openface_outdir, 'reps.csv'), 
                                                          header=None)
    labels = pd.read_csv(os.path.join(self.openface_outdir, 'labels.csv'), 
                                                            header=None)

    #Get the names of each person (by parsing the folder name)
    names = labels.groupby(0).first()[1]
    names = [os.path.dirname(p)[self.output_dir_end:] for p in names]

    #Create the X and y of our dataset
    X = np.array(reps)
    y = labels[0] - 1

    #Encode our labels with LabelEncoder
    self.le = LabelEncoder()
    self.le.fit(names)
    self.n_classes = len(self.le.classes_)

    #Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #Grid search on parameters
    svc = SVC()
    self.clf = GridSearchCV(svc, self.svm_parameters)
    self.clf.fit(X_train, y_train)

    #Predict (note GridSeachCV will automatically use the best one)
    training_predictions = self.clf.predict(X_train)
    training_score = accuracy_score(training_predictions, y_train)
    print(training_score)

    testing_predictions = self.clf.predict(X_test)
    testing_score = accuracy_score(testing_predictions, y_test)
    print(testing_score)


    testing_predictions = self.le.inverse_transform(\
                      testing_predictions.astype('int'))
    y_test = self.le.inverse_transform(y_test.astype('int'))

    for (a, b) in zip(testing_predictions, y_test):
      print("Guessed: {:15} | Answer: {:15}".format(a,b))

    self.clf.fit(X, y)

  #Clear!!
  def clear_directories(self):
    if os.path.exists(self.openface_data_dir):
      shutil.rmtree(self.openface_data_dir)
    if os.path.exists(self.openface_outdir):
      shutil.rmtree(self.openface_outdir)

  #Label the frames!!
  def label_frames(self, frames):
    dump_path = os.path.join(self.output_dir, 'dump/')
    if not os.path.exists(dump_path):
      os.makedirs(dump_path)

    #Keep track of the index and the boxes we've seen in each frame
    index = 0
    box_sets = []
    for frame in frames:
      box_and_image = self.fp.crop_and_align(frame, get_one=False)

      #If we couldn't find a face in the frame, continue...?
      if box_and_image is None:
        continue

      (rects, cropped_images) = box_and_image
      box_sets.append(rects)
      #Write each cropped face to the output directory
      for face in cropped_images:
        output_image_file = os.path.join(dump_path, str(index) + '.jpg')
        cv2.imwrite(output_image_file, face)
        index += 1 
  
    #Call openface lua script
    subprocess.call([self.openface_path, self.openface_outdir_arg, 
                    self.openface_outdir, self.openface_data_dir_arg, 
                    self.openface_data_dir, self.openface_model_arg, 
                    self.openface_model])

    #Get the reps and sort them (so they're in the same order)
    reps = pd.read_csv(os.path.join(self.openface_outdir, 'reps.csv'), 
                                                          header=None)
    labels = pd.read_csv(os.path.join(self.openface_outdir, 'labels.csv'),
                                                          header=None)
    reps[-1] = labels[1]
    reps = reps.sort_values([-1]).drop([-1], axis=1)
    reps = np.array(reps)

    #Iterate through each box_set anx frame...
    labeled_frames = []
    index = 0
    for (box_set, frame) in zip(box_sets, frames):
      if len(box_set):
        #Optimize the predictions (see match_optimize)
        end_index = index + len(box_set)
        probs = self.clf.predict_proba(reps[index:end_index])
        labels = self.match_optimal(probs)
        #Draw the box with the label
        for (text, box) in zip(labels, box_set):
          x, y, w, h = box.left(), box.top(), box.width(), box.height()
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 3)
          cv2.putText(frame, text, (x, y),  
                      cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 0), 3)
        labeled_frames.append(frame)
        index = end_index
      labeled_frames.append(frame)
    assert index == len(reps)
    return labeled_frames

  #Simple greedy algorithm to match each label w/ highest probability
  #Assumption: we shouldn't have the same person more than once
  #if rows > cols => 1 to 1 
  def match_optimal(self, probabilities):
    n = probabilities.shape[0]
    labels = [-1] * n 
    #Match highest probs first; making sure at least 1 of each class
    for _ in range(min(n, self.n_classes)):
      k = probabilities.max(axis=0).argmax()
      index = probabilities[:,k].argmax() 
      labels[index] = k
      probabilities[index] -= 1
      probabilities[:,k] -= 1

    #Match the remainder with highest prob
    if self.n_classes < n:
      for index in range(n):
        if labels[index] == -1:
          labels[index] = probabilities[index].argmax()

    return self.le.inverse_transform(labels)

