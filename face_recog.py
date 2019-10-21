import face_recognition
import os
import numpy as np
import pickle
import time
from multiprocessing import Pool
jobs=[]
key=pickle.load(open('key.pkl','rb'))
known_image_encodings=pickle.load(open('known_image_encodings.pkl','rb'))
for f in os.listdir('./test_criminals'):
    try:
        test=face_recognition.face_encodings(face_recognition.load_image_file('./test_criminals/'+f))[0]
    except:
        continue
    results = face_recognition.compare_faces(known_image_encodings,test,tolerance=0.6)
    results=np.array(results).astype('int')
    k=list(np.where(results==1)[0])
    for i in k:
        print(f,'--> ',key[i])
