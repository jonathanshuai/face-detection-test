import cv2

from framelabeler import FrameLabeler


#Define some filenames, directories, and constants
face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
input_dir = "./images"
output_dir = "./aligned"

#The data and output directories for openface embedding script
openface_path = "../building/openface/batch-represent/main.lua"
openface_outdir = "./reps" 
openface_model = "../building/openface/models/openface/nn4.v2.t7"

BATCH_SIZE = 100

fl = FrameLabeler(face_landmark_dat, input_dir, output_dir,
									openface_path, openface_outdir, openface_model)
fl.clear_directories()
fl.train()
fl.clear_directories()

cap = cv2.VideoCapture('./videos/video_cut3.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./videos/output.avi',fourcc, 35.0, (1920, 1080))

batch = []
batch_length = 0
while(cap.isOpened()):
	_, frame = cap.read()
	batch.append(frame)
	batch_length += 1
	if batch_length >= BATCH_SIZE:
		#stopping condition
		if batch[0] is None:
			break
		for frame in fl.label_frames(batch):
			#cv2.imshow("image_smoothed", frame)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			out.write(frame)
		batch = []
		batch_length = 0
		fl.clear_directories()

cap.release()
out.release()

'''
import cv2
import dlib
import imutils
from imutils import face_utils

import facepreprocessor
from facepreprocessor import FacePreprocessor

face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
fp = FacePreprocessor(face_landmark_dat, size=96)
image_color = cv2.imread("./garbonzo.jpg")


cv2.imshow("image", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, b = fp.crop_and_align(image_color)

for p in b:
	cv2.imshow("image", p)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


_, image_color = fp.crop_and_align(image_color)
image_color = image_color[0]
image_color = fp.apply_gamma_correction(image_color)
image_color = fp.apply_clahe(image_color)
#image_color = fp.apply_smoothing(image_color)

cv2.imshow("image", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''