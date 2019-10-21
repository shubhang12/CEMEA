import cv2
import sys
from PIL import Image
import numpy as np
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
def get_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1,1),
        flags = cv2.CASCADE_SCALE_IMAGE)
    mx=0
    for i in faces:
        if i[2]*i[3]>mx:
            mx=i[2]*i[3]
            (x, y, w, h)=i
    return np.array(Image.fromarray(image).crop((x,y,w+x,h+y)))
def get_face_multi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1,1),
        flags = cv2.CASCADE_SCALE_IMAGE)
    images=[]
    n=0
    for i in faces:
        (x,y,w,h)=i
        img=Image.fromarray(image).crop((x,y,w+x,h+y))
        img.save('test_criminals/test_'+str(n)+'.jpg')
        images.append(np.array(img))
        n+=1

    return np.array(images)
