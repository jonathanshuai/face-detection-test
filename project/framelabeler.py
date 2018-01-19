#Simple class to train on faces, and 
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


import cv2
import dlib
import imutils
from imutils import face_utils

import facepreprocessor
from facepreprocessor import FacePreprocessor


class FrameLabeler:
  def __init__(self, face_landmark_dat, input_dir, output_dir,
              openface_path, openface_outdir, openface_model,
              align_type, min_size=80):
    self.face_landmark_dat = face_landmark_dat #path to landmark.dat that dlib uses for landmark detection
    self.input_dir = input_dir #input training images directory
    self.output_dir = output_dir #output for cropping faces
    self.output_dir_end = len(output_dir) + 1 
    self.input_dir_end = len(input_dir) + 1
    self.openface_path = openface_path #path to openface lua script
    self.openface_outdir_arg = "-outDir" 
    self.openface_outdir = openface_outdir #path for openface to write representations to
    self.openface_data_dir_arg = "-data" 
    self.openface_data_dir = output_dir #path for openface to read cropped face images from
    self.openface_model_arg = "-model"
    self.openface_model = openface_model #path to openface nn model
    self.fp_parameters = {'landmark_dat': face_landmark_dat, 'size': 96, 
                          'alignment': align_type}
    self.min_size = min_size #minimum size of a face 
    #self.model_parameters = [
    #                {'C': [1, 10, 1e2, 1e3, 1e4, 1e5], 
    #                'kernel': ['linear'], 'probability': [True]}, 
    #                {'C': [1, 10, 1e2, 1e3, 1e4, 1e5], 
    #                'gamma': [1e-2, 1e-3, 1e-4],
    #                'kernel': ['rbf'], 'probability': [True]}]
    #self.model = SVC()
    self.model_parameters = {'C': [1, 10, 1e2, 1e3, 1e4, 1e5], 
                            'penalty': ['l1', 'l2']} 
    self.model = LogisticRegression()
    #self.model_parameters = {'n_neighbors': [1, 3, 5, 7]}
    #self.model = KNeighborsClassifier()
    self.fp = FacePreprocessor(**self.fp_parameters)
    self.preprocess_pipeline = [self.fp.apply_clahe] 
    #                            self.fp.apply_smoothing, 
    #                            self.fp.apply_gamma_correction]
    #self.preprocess_pipeline = []


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

        (_, image_cropped) = box_and_image
        #Since crop_and_align returns an array, get the first one
        image_cropped = image_cropped[0]
        #Preprocess 
        for p in self.preprocess_pipeline:
          image_cropped = p(image_cropped)
        #cv2.imshow("image", image_cropped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

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
    labels = [os.path.dirname(p)[self.output_dir_end:] for p in labels[1]]

    #Encode our labels with LabelEncoder
    self.le = LabelEncoder()
    self.le.fit(labels)
    self.n_classes = len(self.le.classes_)

    #Create the X and y of our dataset
    X = np.array(reps)
    y = self.le.transform(labels)

    #Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #Grid search on parameters
    self.clf = GridSearchCV(self.model, self.model_parameters)
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
        box_sets.append([])
        continue

      (rects, cropped_images) = box_and_image
      box_set = []
      #Write each cropped face to the output directory
      for (rect, image_cropped) in zip(rects, cropped_images):
        if rect.width() > self.min_size:
          #Preprocess 
          for p in self.preprocess_pipeline:
            image_cropped = p(image_cropped)
          
          output_image_file = os.path.join(dump_path, str(index) + '.jpg')
          cv2.imwrite(output_image_file, image_cropped)
          index += 1 
          box_set.append(rect)

      box_sets.append(box_set)
  
    if index == 0:
      return frames

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

    predictions = self.le.inverse_transform(self.clf.predict(reps))

    #Iterate through each box_set and frame...
    labeled_frames = []
    index = 0
    for (box_set, frame) in zip(box_sets, frames):
      if len(box_set):
        #Optimize the predictions (see match_optimize)
        end_index = index + len(box_set)
        labels = predictions[index:end_index]
        #Draw the box with the label
        for (text, box) in zip(labels, box_set):
          x, y, w, h = box.left(), box.top(), box.width(), box.height()
          cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
          cv2.putText(frame, text, (x, y+h),  
                      cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
        labeled_frames.append(frame)
        index = end_index
      labeled_frames.append(frame)
    assert index == len(reps)
    return labeled_frames

  #Simple greedy algorithm to match each label w/ highest probability
  #Assumption: we shouldn't have the same person more than once
  #if rows > cols => 1 to 1 
  #Decided not to use this because it is WEIRD!! (also doesn't work well with KNN and SVM)
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

    print(probabilities)
    print(labels)
    return self.le.inverse_transform(labels)

