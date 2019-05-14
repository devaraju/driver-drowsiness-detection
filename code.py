from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2

pygame.mixer.init() #Initialize Pygame and load music
pygame.mixer.music.load('files/red_alert.wav')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return((A+B) / C);

def drawPoints(eye):
	for cent in eye:
		cv2.circle(frame,tuple(cent), 1, (255,255,255), 1);

###################
EAR_THRESHOLD = 0.52
EAR_FRAMES = 15
COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('files/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cam = cv2.VideoCapture(0)
time.sleep(1)

while(True):
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    cv2.putText(frame, "Press 'q' to QUIT", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1)
    faces = detector(gray, 0) #Detect facial points through detector function
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape);
        leftEye = shape[lStart:lEnd] #Get array of coordinates of leftEye and rightEye
        rightEye = shape[rStart:rEnd];
        drawPoints(leftEye);
        drawPoints(rightEye);
        
        l_EAR = eye_aspect_ratio(leftEye)
        r_EAR = eye_aspect_ratio(rightEye)
        EAR = (l_EAR + r_EAR) / 2;
        
        if(EAR < EAR_THRESHOLD):
            COUNTER += 1;
            ear_str = str(EAR);
            cv2.putText(frame, ("Eye Aspect Ratio:"+ear_str[:4]+"*"), (10,420), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,0,255), 1);
            if COUNTER >= EAR_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "ALERT: Drowsiness detected...", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2);
                
        else:
            pygame.mixer.music.stop()
            COUNTER = 0;
        ear_str = str(EAR);
        cv2.putText(frame, ("Eye Aspect Ratio:"+ear_str[:4]), (10,420), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (222,0,0), 1);

    cv2.imshow('Drowsiness Detection System', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
