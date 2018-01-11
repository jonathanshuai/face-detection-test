import warnings

import numpy as np

import cv2
import imutils
from imutils import face_utils
import dlib
import openface

#Sources:
#https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
#https://www.researchgate.net/publication/239084542_Evaluation_of_Image_Pre-Processing_Techniques_for_Eigenface_Based_Face_Recognition

class FacePreprocessor:
  #Initialize parameters and face recognizers
  def __init__(self, landmark_dat, left_eye_pos=(0.32, 0.32), 
                  width=300, height=300, clahe_clip_limit=2.0,
                  clahe_tile_grid_size=(8, 8)):
    self.left_eye_position = left_eye_pos
    self.width = width
    self.height = height
    self.clahe_clip_limit = clahe_clip_limit
    self.clahe_tile_grid_size = clahe_tile_grid_size

    #For face detection and alignment
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(landmark_dat)

    #For normalizing brightness
    self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                 tileGridSize=self.clahe_tile_grid_size)


  def crop_and_align(self, image_color):
    if image_color.ndim != 3:
      raise ValueError('Image format incorrect (was not RGB)')

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    faces = self.detector(image_gray, 0)

    cropped_images = []
    for (i, face) in enumerate(faces):
      # get facial landmark shapes (and turn into )
      shape = self.predictor(image_gray, face)
      shape = face_utils.shape_to_np(shape)
     
      #Get the eye centers
      (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
      (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

      left_eye_points = shape[left_start:left_end]
      right_eye_points = shape[right_start:right_end]

      left_eye_center = left_eye_points.mean(axis=0).astype("int")
      right_eye_center = right_eye_points.mean(axis=0).astype("int")

      #Find the angle between the eyes
      dX = right_eye_center[0] - left_eye_center[0]
      dY = right_eye_center[1] - left_eye_center[1]
      angle = np.degrees(np.arctan2(dY, dX)) - 180

      #Find the scaling factor (scaled distance / actual distance between eyes)
      right_eye_x = 1.0 - self.left_eye_position[0]
      scaled_distance = self.width * (right_eye_x - self.left_eye_position[0])
      actual_distance = np.sqrt(dX ** 2 + dY ** 2)
      scale = scaled_distance / actual_distance

      #Midpoint between eyes
      eye_midpoint = ((right_eye_center[0] + left_eye_center[0]) // 2, 
                      (right_eye_center[1] + left_eye_center[1]) // 2)

      #Get rotation matrix (rotate about eye_midpoint)
      M = cv2.getRotationMatrix2D(eye_midpoint, angle, scale)
      #make photo top left corner starts at eye_midpoint
      M[0, 2] -= eye_midpoint[0]
      M[1, 2] -= eye_midpoint[1]
      #then, add the offset to include the rest of the face 
      M[0, 2] += self.width * 0.5
      M[1, 2] += self.height * self.left_eye_position[1]

      image_cropped = cv2.warpAffine(image_color, M, (self.width, self.height))
      cropped_images.append(image_cropped)

    return cropped_images


  #Turn image to gray (do nothing if already gray)
  def rgb_to_gray(self, image_color):
    if image_color.ndim == 2:
      warnings.warn(\
        'Image was not RGB; returning the same image', UserWarning)

      return image_color

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    return image_gray

  #To normalize areas with high brightness 
  def apply_clahe(self, image_color):
    image_gray = self.rgb_to_gray(image_color)
    return self.clahe.apply(image_gray)

  #Smooth out using gaussian blur
  def apply_smoothing(self, image, size=(5, 5), 
                        sig_x=0, sig_y=0):
    image_smoothed = cv2.GaussianBlur(image, size, 
                                      sig_x, sig_y)
    return image_smoothed

  #Canny edge detection
  def apply_canny(self, image, min_val=100, max_val=200):
    image_edges = cv2.Canny(image, min_val, max_val)
    return image_edges
  

  #Create a lookup table for gamma correction
  #Note: formula for gamma correction is 
  # (image/255 * 1/gamma) * 255
  def build_gamma_table(self, gamma=1.5):
    invGamma = 1.0 / gamma
    self.gamma_table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   
  #Apply gamma correction (does this actually do anything...?)
  def apply_gamma_correction(self, image):
    try:
      self.gamma_table
    except:
      warnings.warn('No gamma table was created. \
        Creating one with default gamma value 1.5', UserWarning)

      self.build_gamma_table()

    image_gamma = cv2.LUT(image, self.gamma_table)
    return image_gamma

  def apply_dog(self, image, size1=(3,3), sig_x1=0, sig_y1=0,
                size2=(5,5), sig_x2=0, sig_y2=0):
    g1 = cv2.GaussianBlur(image, size1, sig_x1, sig_y1)
    g2 = cv2.GaussianBlur(image, size2, sig_x2, sig_y2)
    image_dog = g1-g2
    return image_dog