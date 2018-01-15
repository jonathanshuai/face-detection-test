import warnings

import numpy as np

import cv2
import imutils
from imutils import face_utils
import dlib

import openface
import openface.helper
from openface.data import iterImgs

#Sources:
#https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
#https://www.researchgate.net/publication/239084542_Evaluation_of_Image_Pre-Processing_Techniques_for_Eigenface_Based_Face_Recognition

#Note: Increasing left_eye_pos will "zoom out" on the face, whereas 
#increasing lip_pos will "squeeze" the face vertically
#Defaults for left_eye_pos and third_pos should be good enough in most cases
#face landmark indices to align the inner eyes and the bottom of the lip
LEFT_INNER_EYE = 39
RIGHT_INNER_EYE = 42
BOTTOM_LIP = 57

#face landmark indices for outer eyes and the nose (tip)
LEFT_OUTER_EYE = 36
RIGHT_OUTER_EYE = 45 
NOSE_TIP = 33

LIP = 0
NOSE = 1

ALIGNMENT_POINTS = [[LEFT_INNER_EYE, RIGHT_INNER_EYE, BOTTOM_LIP],
                    [LEFT_OUTER_EYE, RIGHT_OUTER_EYE, NOSE_TIP]]

MARGIN = 0.25
class FacePreprocessor:
  #Initialize parameters and face recognizers
  def __init__(self, landmark_dat, alignment=LIP, size=200,
              left_eye_pos=None, third_pos=None):

    self.alignment = alignment
    self.size = size
    
    #Set the indicies based on alignment type
    self.alignment_indices = ALIGNMENT_POINTS[alignment]

    #For face detection and alignment
    self.detector = dlib.get_frontal_face_detector()
    self.aligner = openface.AlignDlib(landmark_dat)


  def crop_and_align(self, image_color, get_one=True):
    faces = self.crop(image_color, get_one=get_one)
    return self.align(image_color, faces)

  def crop(self, image_color, get_one=True):
    if image_color is None:
      return None

    if image_color.ndim != 3:
      raise ValueError('Image format incorrect (was not RGB)')

    faces = self.detector(image_color, 1)
    if not faces:
      return None

    if get_one:
      #Get the largest face
      sizes = [rect.width() * rect.height() for rect in faces]
      faces = [faces[int(np.argmax(sizes))]]

    return faces

  def align(self, image_color, faces):
    if not faces:
      return None

    cropped_faces = []
    rects = []
    for face in faces: 
      left, top, right, bottom = self.get_bounds(face, image_color)

      crop = image_color[top:bottom, left:right]
      rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
      #cv2.imshow("image", crop)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()
      outRgb = self.aligner.align(self.size, rgb,
                            landmarkIndices=self.alignment_indices,
                            skipMulti=False)
      if not outRgb is None:
        rgb = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
        cropped_faces.append(rgb)
        rects.append(face)

      else:
        warnings.warn("Could not align rectangle...!!", UserWarning)

    if not len(rects):
      return None

    return rects, cropped_faces


  def get_bounds(self, face, image):
    (max_height, max_width) = image.shape[:2]
    x_margin = int(face.width() * MARGIN)
    y_margin = int(face.height() * MARGIN)
    left = max(0, face.left() - x_margin)
    top = max(0, face.top() - y_margin)
    right = min(max_width, face.right() + x_margin)
    bottom = min(max_height, face.bottom() + y_margin)
    return left, top, right, bottom

  #Turn image to gray (do nothing if already gray)
  def bgr_to_gray(self, image_color):
    if image_color.ndim == 2:
      warnings.warn(\
        'Image was not RGB; returning the same image', UserWarning)

      return image_color

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    return image_gray


  def create_clahe(self, clahe_clip_limit=2.0,
                  clahe_tile_grid_size=(2, 2)):
    #For normalizing brightness
    self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                 tileGridSize=clahe_tile_grid_size)

  #To normalize areas with high brightness 
  def apply_clahe(self, image_color):
    try:
      self.clahe
    except:
      warnings.warn(\
        'No CLAHE was created. Creating one with default parameters', UserWarning)
      self.create_clahe()

    if image_color.ndim == 2:
      image_gray = self.bgr_to_gray(image_color)
      return self.clahe.apply(image_gray)
    else:
      lab = cv2.cvtColor(image_color, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab)
      cl = self.clahe.apply(l)
      limg = cv2.merge((cl,a,b))
      final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
      return final



  #Smooth out using gaussian blur (size and sig values from Heseltine et al.)
  #For more smoothing: size=(5, 5), sig_x=1.028, sig_y=1.028
  def apply_smoothing(self, image, size=(3, 3), 
                        sig_x=0.788, sig_y=0.788):
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

  #Apply dog (difference of gaussians) 
  def apply_dog(self, image, size1=(3,3), sig_x1=0, sig_y1=0,
                size2=(5,5), sig_x2=0, sig_y2=0):
    g1 = cv2.GaussianBlur(image, size1, sig_x1, sig_y1)
    g2 = cv2.GaussianBlur(image, size2, sig_x2, sig_y2)
    image_dog = g1-g2
    return image_dog