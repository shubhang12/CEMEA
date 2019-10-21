from senti import detect_emotion
from face_detection import *
import cv2
import numpy as np
import time
image=cv2.imread('4.jpg')
for i in get_face_multi(image):
    print(detect_emotion(i))
