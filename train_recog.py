import face_recognition
import os
import numpy as np
import pickle
import time
from multiprocessing import Pool
t=time.time()
job=[]
for f in os.listdir('./known_criminals'):
    job.append('./known_criminals/'+f)
n=0
def fun(j):
    global n
    print(n,end='\r')
    n+=12
    try:
        enc=face_recognition.face_encodings(face_recognition.load_image_file(j))[0]
        key=f[:-4]
        return [enc,key]
    except:
        pass

if __name__=='__main__':
    key=[]
    known_image_encodings=[]
    p=Pool()
    res=p.map(fun,job[:])
    for i in res:
        if i!=None:
            known_image_encodings.append(i[0])
            key.append(i[1])
    key=np.array(key)
    pickle.dump(key,open('key.pkl','wb'))
    pickle.dump(known_image_encodings,open('known_image_encodings.pkl','wb'))
