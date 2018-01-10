import time 
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import cv2
import imutils


def rotate_points(x_coords, y_coords, shape, theta):
  n = len(x_coords)
  if n != len(y_coords):
    raise ValueError("The number of x and y coordinates are not equal.") #wrap?
  if n == 0:
    raise ValueError("Coordinate lists are empty")

  (height, width) = shape
  (centerX, centerY) = (width // 2, height // 2)

  #Transform each corner of the rectangle
  M = cv2.getRotationMatrix2D((centerX, centerY), theta, 1.0)
  shift = M[:, 2]
  rect = np.array([x_coords, y_coords, [1] * n ])
  new_rect = np.dot(M, rect)

  #Format the points to draw polylines 
  pts = [[new_rect[0, i], new_rect[1, i]] for i in range(-1, n)]
  pts = np.array(pts, np.int32)
  pts.reshape((-1, 1, 2))
  return pts

def point_angle_line(x, y, theta, low, high):
  assert high >= y and low <= y, "y should be in the bounds of the image"

  #Calculate the lengths from the change in y
  length_1 = (low - y) / np.sin(theta)
  length_2 = (high - y) / np.sin(theta)
  
  #Calculate delta_x's 
  delta_x_1 = int(length_1 * np.cos(theta))
  delta_x_2 = int(length_2 * np.cos(theta))
  
  #Return new points
  p1 = (x + delta_x_1, low)
  p2 = (x + delta_x_2, high)  
  return p1, p2

#Define some filenames and constants
csv_file = "data.csv"
lbp_xml = "./sets/lbpcascade_frontalface_improved.xml"
eye_xml = "./sets/haarcascade_eye.xml"
left_eye_xml = "./sets/haarcascade_lefteye_2splits.xml"
right_eye_xml = "./sets/haarcascade_righteye_2splits.xml"
CROP_SIZE = 160

jisoo_id = "js"
seolhyun_id = "sh"
songhyekyo_id = "shk"
hanhyojoo_id = "hhj"

#Read in the data to a pandas dataframe
df = pd.read_csv(csv_file)

#Get only Jisoo and Seolhyun images
df = df.loc[(df['id'] == jisoo_id) | (df['id'] == seolhyun_id)]
#df = df.loc[(df['id'] == jisoo_id) | (df['id'] == songhyekyo_id)]
#df = df.loc[(df['id'] == hanhyojoo_id) | (df['id'] == songhyekyo_id)]
#df = df.loc[(df['id'] == jisoo_id)]
#df = df.loc[(df['id'] == seolhyun_id)]
df = df.sort_values('path')

#df = df.loc[(df['path'] == './images/jisoo/jisoo13.jpg')]

df2 = pd.DataFrame(columns=['face', 'label'])

#Iterate through each picture
for i, r in df.iterrows():
  image_file = r['path']

  #read image in grayscale (flag 0)
  img_c = cv2.imread(image_file)
  img_g = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

  (height, width) = img_g.shape[:2]

  #Keep track of whether we found a face or not
  found_face = False

  for theta in [0, 15, -15, 30, -30]:
    rotated = imutils.rotate(img_g, theta)

    face_cascade = cv2.CascadeClassifier(lbp_xml)
    faces = face_cascade.detectMultiScale(
      rotated,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30)
    )

    if len(faces):
      found_face = True

      for (fx, fy, fw, fh) in faces:
       
        cropped = rotated[fy:fy+fh, fx:fx+fw] #crop face

        eye_cascade = cv2.CascadeClassifier(eye_xml)
        eyes = eye_cascade.detectMultiScale(cropped, minSize=(10, 10))

        #If we found (at least) a pair of eyes...
        if len(eyes) >= 2:
          #Sort eyes by the y value 
          eyes = eyes[eyes[:,1].argsort()]

          #Get the highest two points -- these should be the actual eyes;
          #if more are detected, they're usually nose holes or in the teeth
          left_eye = eyes[0]
          right_eye = eyes[1]
          #Make sure the left eye is on the left
          if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye

          #Draw a circle in each eye (in color)
          eye_centers = []
          for (eye_x, eye_y, eye_w, eye_h) in (left_eye, right_eye):
            eye_x, eye_y = eye_x + fx, eye_y + fy #add offset from face

            #Get the centers of each eye
            center_x = eye_x + (eye_w // 2)
            center_y = eye_y + (eye_h // 2)
            eye_centers.append((center_x, center_y))

            #Draw circles in the eyes
            #cv2.circle(rotated, (center_x, center_y), 4, (255, 0, 0), -1)
           
            x_coords = [center_x]
            y_coords = [center_y]
            pts = rotate_points(x_coords, y_coords, img_g.shape[:2], -(theta))
            rotated_x = pts[0, 0]
            rotated_y = pts[0, 1]
            cv2.circle(img_c, (rotated_x, rotated_y), 4, (200, 0, 200), -1)

          
          #Find the angle between the eyes
          (cx_l, cy_l) = eye_centers[0]
          (cx_r, cy_r) = eye_centers[1]
          slope = (cy_r - cy_l) / (cx_r - cx_l)
          align_angle = np.arctan(slope)

          #Calculate the midpoint between the eyes
          mid_x = (cx_l + cx_r) // 2
          mid_y = (cy_l + cy_r) // 2

          #Draw a line using the angle perpendicular to the alignment angle
          perp_angle = align_angle + (np.pi/2)
          p1, p2 = point_angle_line(mid_x, mid_y, perp_angle, fy, fy+fh)
          #cv2.line(rotated, p1, p2, (0, 255, 0), 2)

          #Get angle in degrees
          align_angle_deg = np.rad2deg(align_angle)

          #Rotate the image to align
          rotated = imutils.rotate(rotated, align_angle_deg)

          #Rotate the box about the center of the image
          x_coords = [0, 0, fw, fw] + fx
          y_coords = [0, fh, fh, 0] + fy
          pts = rotate_points(x_coords, y_coords, img_g.shape[:2], align_angle_deg)
          
          #Rotate the box about the center of the box
          x_coords, y_coords = pts[:,0], pts[:,1]
          new_fx, new_fy = min(x_coords), min(y_coords)
          x_coords -= new_fx
          y_coords -= new_fy
          pts = rotate_points(x_coords, y_coords, (max(y_coords), max(x_coords)), -align_angle_deg)
          pts[:,0] += new_fx
          pts[:,1] += new_fy

          #Crop the photo to the aligned box
          lower_x, upper_x = min(pts[:,0]), max(pts[:,0])
          lower_y, upper_y = min(pts[:,1]), max(pts[:,1])
          crop_width = upper_x - lower_x
          crop_height = upper_y - lower_y
          upper_x -= crop_width - crop_height #make it a square
          cropped = rotated[lower_y:upper_y, lower_x:upper_x]

          #Draw a rectangle around the colored face (rotate in place; then rotate wrt image)          
          x_coords = [0, 0, fw, fw] 
          y_coords = [0, fh, fh, 0] 
          pts = rotate_points(x_coords, y_coords, (fh, fw), -align_angle_deg)
          pts[:,0] += fx
          pts[:,1] += fy
          x_coords, y_coords = list(pts[:-1,0]), list(pts[:-1,1])
          pts = rotate_points(x_coords, y_coords, img_g.shape[:2], -theta)
          cv2.polylines(img_c, [pts], False, (0, 255, 0), 2)

          #Draw the line separating the face
          x_coords = [p1[0], p2[0]]
          y_coords = [p1[1], p2[1]]
          pts = rotate_points(x_coords, y_coords, img_g.shape[:2], -theta)
          x_coords, y_coords = pts[:-1,0], pts[:-1,1]
          p1 = (x_coords[0], y_coords[0])
          p2 = (x_coords[1], y_coords[1])
          cv2.line(img_c, p1, p2, (255, 0, 0), 2)

      #cropped = cv2.resize(cropped, (CROP_SIZE, CROP_SIZE))

      #Show the colored face with markers
      #cv2.imshow(r['path'], img_c)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()

      #Show the cropped face
      #cv2.imshow(r['path'], cropped)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()

      cropped = cv2.resize(cropped, (CROP_SIZE, CROP_SIZE))

      #corrected = cv2.equalizeHist(cropped)
      clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2,2))
      corrected = clahe.apply(cropped)


      #Show the histogram equalization  
      cv2.imshow("corrected", corrected)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      df2 = df2.append({'face': corrected.ravel(), 'label': r['label']}, ignore_index=True)

      break

  if not found_face:
    warnings.warn("Could not find face in {}".format(r['path']), RuntimeWarning)

