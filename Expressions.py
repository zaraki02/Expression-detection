#!/usr/bin/env python
# coding: utf-8

# # Detecting Expressions...

# In[1]:


import numpy
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from keras.models import load_model


# In[2]:


img1 = cv2.imread(r"C:\Users\abhij\Downloads\angry.jpg")
img2 = cv2.imread(r"D:\my\IMG_20190304_231507.jpg")
img3 = cv2.imread(r"C:\Users\abhij\Downloads\upset2.jpg")

#gray_img1 = cv2.imread(r"D:\my\IMG_20190611_131516.jpg",0)
#gray_img2 = cv2.imread(r"D:\my\IMG_20190304_231507.jpg",0)

classifier = cv2.CascadeClassifier(r"C:\Users\abhij\DATA\haarcascades\haarcascade_frontalface_default.xml")

model = load_model("exp.h5")
def detected_face(img):
    
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coord = classifier.detectMultiScale(img,1.1,5)
    #print([coord[0]])
    for x,y,w,h in coord:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        face = img[y:y+h,x:x+w]
        resized = cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)
        face_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        face_gray = numpy.reshape(face_gray,[face_gray.shape[0],face_gray.shape[1],1])
        face_gray = face_gray/255
        print(face_gray.shape)
        expand_face_gray = numpy.expand_dims(face_gray,axis=0)
        print(expand_face_gray.shape)
        result = model.predict_classes(expand_face_gray)
        #Creating a dict for mapping the numpy values...
        emotions = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
        percentage = max(model.predict(expand_face_gray))*100
        print(percentage)
        print(face.shape)
        
        res=str(emotions[result[0]])+str(percentage)
        emo =emotions[result[0]]
        cv2.putText(img,emo,org=(x,y-10),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale =1,color=(255,255,255),thickness=1)
        #plt.imshow(face)
        #plt.show()
    #plt.imshow(img)
    #plt.show()


# In[15]:


detected_face(img1)
detected_face(img2)
detected_face(img3)


# In[3]:


cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read(0)
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face(frame)
    
    
    cv2.imshow("MyFrame",frame)
    k = cv2.waitKey(2)
    if k==27 or k==ord("e"):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




