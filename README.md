# emotion_detection
# Introduction
What is facial emotion recognition? Facial emotion recognition is the process of detecting human emotions from facial expressions. The human brain recognizes emotions automatically, and software has now been developed that can recognize emotions as well.
# Steps
# set enevironments
!pip3 install virtualenv
!virtualenv theanoEnv
# Activate
!source /content/theanoEnv/bin/activate theanoEnv

clone the repo https://github.com/Snehadhole/emotion_detection

cd emotion_detection

!pip install -r requirements.txt

!pip install tensorflow-gpu==2.0.0

cd /content/Emotion-detection/src

folder structure :
Emotion-detection

     src
	
       new folder
	  
       dataset_prepare.py
	  
       emotions.py
	  
       haarcascade_frontalface_default.xml
	  
       model.h5
	  
make all changes to emotions.py

!python emotions.py --mode display

# To resize the video size
!ffmpeg -i  img path -vf scale=240:320 o/p img path
i.e. !ffmpeg -i /content/all.mp4 -vf scale=240:320 all1.mp4
