## Notes
This repo was an exploration with face detection and recognition. I started using OpenCV’s implementation of face detection using Haar Cascades and [Viola-Jones](https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid) facial detection algorithm. After detecting and cropping the faces, I tried to train a classifier based on the eigenface model (using PCA and SVM) to recognize faces in a testing set. After aligning the face (using [dlib](https://github.com/davisking/dlib)’s facial landmark predictor) and preprocessing (applying smoothing and other techniques mentioned [in this paper](https://www.researchgate.net/publication/239084542_Evaluation_of_Image_Pre-Processing_Techniques_for_Eigenface_Based_Face_Recognition) by Heseltine from the University of New York). I got an okay accuracy with this model, ~70%-80%. 

I put the old files for eigen face, fisher face, and the face preprocessor in the old_files directory.

Looking around for other models, I learned about Google’s [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) model, and an api, [openface](https://github.com/cmusatyalab/openface), that implements this model. After downloading and installing from the repo (I put the folder in ./building/openface/ directory). Basically it uses a trained cnn (which openface provides) to turn each face into a set of 128 features. In the FaceNet paper, they used knn to classify the faces. I tried knn, logistic regression, and svm ~ they all had similar performance..?

I tried training this model and running it on each frame of a music video, to see if it could identify members of a group. Here’s a clip from the results from training on One Direction music videos (more on this later) ‘Best Song Ever’ and ‘You and I’ to recognize the members in ‘Story of My Life’:


![alt text](https://github.com/jonathanshuai/face-detection-test/blob/master/project/records/story_of_my_life.gif?raw=true)

It worked well, but sometimes it seemed to get really confused (jumping between labels). I think it’s because sometimes if the face is blurred from moving it could make a bad prediction for that frame. Also, small/low resolution faces come out poorly.

![alt text](https://raw.githubusercontent.com/jonathanshuai/face-detection-test/master/project/records/25.jpg?token=AK2CHnqzMIcsFqBkOwDgJXaARPox6hHaks5aatzpwA%3D%3D)
![alt text](https://raw.githubusercontent.com/jonathanshuai/face-detection-test/master/project/records/29.jpg?token=AK2CHjg4Xa7_X8QGNVdcQ6Y3xsWHfSdyks5aat0RwA%3D%3D)

This could probably be improved by some kind of smoothing algorithm (maybe voting on the faces for some window of frames?)


As a side note, at first, I built the training set by hand by searching through Google images. But I found that a better way was to get train on faces from other music videos. I wrote a simple get_video_faces.py that will crop out faces from each frame of a video; making it much faster to get a large number of faces with different facial expressions

* Performed well on 'good' pictures (high quality face shots) ~90-95% accuracy on training sets.
* Better than humans for some sets (I'll update to explain this more soon)
* I just used the pipeline that openface provides for classification; but this is slow because it involves writing each face to disk. 
* For labeling faces on music videos, the flickering of the label boxes is not smooth and looks choppy. Doesn't perform well when images are small or blurred.

## Acknowledgments

* [dlib](https://github.com/davisking/dlib)
* [openface](https://github.com/cmusatyalab/openface)
* [OpenCV](https://opencv.org/)
