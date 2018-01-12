import warnings

import numpy as np

import cv2
import imutils
from imutils import face_utils
import dlib

#Sources:
#https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
#https://www.researchgate.net/publication/239084542_Evaluation_of_Image_Pre-Processing_Techniques_for_Eigenface_Based_Face_Recognition


#face landmark indices to align the inner eyes and the bottom of the lip
LEFT_INNER_EYE = 39
RIGHT_INNER_EYE = 42
BOTTOM_LIP = 57

#
LEFT_OUTER_EYE = 36
RIGHT_OUTER_EYE = 45 
NOSE = 33

#Note: Increasing left_eye_pos will "zoom out" on the face, whereas 
#increasing lip_pos will "squeeze" the face vertically
class FacePreprocessor:
  #Initialize parameters and face recognizers
  def __init__(self, landmark_dat, left_eye_pos=(0.37, 0.37), lip_pos=0.82,
                  width=200, height=200):
    self.left_eye_position = left_eye_pos
    self.lip_position = (0.5, lip_pos)
    self.width = width
    self.height = height

    #For face detection and alignment
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(landmark_dat)

  def crop_and_align(self, image_color, get_one=True):
    if image_color.ndim != 3:
      raise ValueError('Image format incorrect (was not RGB)')

    #Detect face using dlib's face detector
    faces = self.detector(image_color, 1)
    if not faces:
      return None

    if get_one:
      #Get the largest face
      sizes = [rect.width() * rect.height() for rect in faces]
      faces = [faces[int(np.argmax(sizes))]]
    
    cropped_faces = []
    for face in faces:
      shape = self.predictor(image_color, face)
      landmarks = face_utils.shape_to_np(shape)
     
      #Get the coords points for eyes and lips
      left_eye_center = landmarks[39]
      right_eye_center = landmarks[42]
      lip_center = landmarks[57]

      #Find the offest from center of picture, and the 
      #(0, 0) anchor position relative to the unscaled image
      offset_x = self.left_eye_position[0] * self.width
      offset_y = self.left_eye_position[1] * self.height
      anchor_x = left_eye_center[0] - offset_x 
      anchor_y = left_eye_center[1] - offset_y
        
      #Calculate the scaled points relative to unscaled image
      scaled_left_eye = [self.left_eye_position[0] * self.width + anchor_x, 
                          self.left_eye_position[1] * self.height + anchor_y]
        
      scaled_right_eye = [(1 - self.left_eye_position[0]) * self.width + anchor_x, 
                          (self.left_eye_position[1]) * self.height + anchor_y]
        
      scaled_lip= [self.lip_position[0] * self.width + anchor_x,
                    self.lip_position[1] * self.height + anchor_y]

      src_triangle = np.float32([left_eye_center, right_eye_center, lip_center])
      scaled_triangle = np.float32([scaled_left_eye, scaled_right_eye, scaled_lip])

      #Get transformation matrix based off the 3 points
      M = cv2.getAffineTransform(src_triangle, scaled_triangle)

      #Change the transformation matrix to start at left eye
      M[0, 2] -= left_eye_center[0]
      M[1, 2] -= left_eye_center[1]
      #Then, add the offset to include the rest of the face 
      M[0, 2] += offset_x
      M[1, 2] += offset_y

      image_cropped = cv2.warpAffine(image_color, M, (self.width, self.height))
      
      cropped_faces.append(image_cropped)

    return cropped_faces
      

  #Turn image to gray (do nothing if already gray)
  def rgb_to_gray(self, image_color):
    if image_color.ndim == 2:
      warnings.warn(\
        'Image was not RGB; returning the same image', UserWarning)

      return image_color

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    return image_gray


  def create_clahe(self, clahe_clip_limit=2.0,
                  clahe_tile_grid_size=(8, 8)):
    #For normalizing brightness
    self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                 tileGridSize=self.clahe_tile_grid_size)

  #To normalize areas with high brightness 
  def apply_clahe(self, image_color):
    image_gray = self.rgb_to_gray(image_color)
    try:
      self.clahe
    except:
      warnings.warn(\
        'No CLAHE was created. Creating one with default parameters', UserWarning)
      self.create_clahe

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