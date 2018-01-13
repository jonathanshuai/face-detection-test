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


fl = FrameLabeler(face_landmark_dat, input_dir, output_dir,
									openface_path, openface_outdir, openface_model)

fl.train()
fl.clear_directories()

image_color = cv2.imread('garbonzo.jpg')
fl.label_frames([image_color])


