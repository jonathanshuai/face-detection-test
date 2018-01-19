#This is a simple script to get faces from a video 
import os
import cv2
import dlib
import imutils
from imutils import face_utils

import facepreprocessor
from facepreprocessor import FacePreprocessor


#Define some filenames, directories, and constants
face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
input_dir = "./images"
output_dir = "./video_crops"

INCREMENT = 10
NUMBER_OF_FRAMES = 10000
MIN_SIZE = 80

fp = FacePreprocessor(face_landmark_dat, size=96)
cap = cv2.VideoCapture('./videos/stay.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

n = 0
index = 0
batch = []
while(cap.isOpened()):
  _, frame = cap.read()
  
  if frame is None:
    break

  if index % INCREMENT == 0:
    batch.append(frame)
    index = index // INCREMENT
    n += 1
    if n >= NUMBER_OF_FRAMES:
      break 

  index += 1

cap.release()


dump_path = os.path.join(output_dir, 'dump/')
if not os.path.exists(dump_path):
  os.makedirs(dump_path)
index = 0
for frame in batch:
  if frame is None:
    break

  boxes = fp.crop(frame, get_one=False)
  if not boxes is None:
    for box in boxes:
      if box.width() < MIN_SIZE:
        continue

      (left, top, right, bottom) = fp.get_bounds(box, frame)
      #Write the image to the designated diretory
      output_image_file = os.path.join(dump_path, str(index) + 'a.jpg')
      cv2.imwrite(output_image_file, frame[top:bottom, left:right])
      index += 1
