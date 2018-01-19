import cv2
import facepreprocessor
from framelabeler import FrameLabeler


#Define some filenames, directories, and constants
face_landmark_dat = "./models/shape_predictor_68_face_landmarks.dat"
input_dir = "./images"
output_dir = "./aligned"

#The data and output directories for openface embedding script
openface_path = "../building/openface/batch-represent/main.lua"
openface_outdir = "./reps" 
openface_model = "../building/openface/models/openface/nn4.v2.t7"
align_type = facepreprocessor.LIP

BATCH_SIZE = 100

fl = FrameLabeler(face_landmark_dat, input_dir, output_dir,
									openface_path, openface_outdir, openface_model,
									align_type)
fl.clear_directories()
fl.train()
fl.clear_directories()

cap = cv2.VideoCapture('./videos/whistle.mp4')
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