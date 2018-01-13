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

BATCH_SIZE = 50

fl = FrameLabeler(face_landmark_dat, input_dir, output_dir,
									openface_path, openface_outdir, openface_model)
fl.clear_directories()
fl.train()
fl.clear_directories()

cap = cv2.VideoCapture('./videos/video_cut.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./videos/output.avi',fourcc, 20.0, (1280,720))

batch = []
batch_length = 0
while(cap.isOpened()):
	_, frame = cap.read()
	batch.append(frame)
	batch_length += 1
	if batch_length >= BATCH_SIZE:
		for frame in fl.label_frames(batch):
			cv2.imshow("image_smoothed", frame)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			out.write(frame)
		batch = []
		batch_length = 0
		fl.clear_directories()

cap.release()
out.release()

cv2.imshow("image_smoothed", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
