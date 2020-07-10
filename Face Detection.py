#!/usr/bin/env python
# coding: utf-8

# # Face Detection...

# In[1]:


import numpy
import matplotlib.pyplot as plt
import pandas
import cv2


# In[20]:


img1 = cv2.imread(r"D:\my\IMG_20190611_131516.jpg")
img2 = cv2.imread(r"C:\Users\abhij\OneDrive\Pictures\Screenshots\Screenshot (36).png")

#gray_img1 = cv2.imread(r"D:\my\IMG_20190611_131516.jpg",0)
#gray_img2 = cv2.imread(r"D:\my\IMG_20190304_231507.jpg",0)

classifier = cv2.CascadeClassifier(r"C:\Users\abhij\DATA\haarcascades\haarcascade_frontalface_default.xml")

def detected_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coord = classifier.detectMultiScale(gray_img,1.2,6)
    return coord


# In[25]:


img2 = cv2.imread(r"C:\Users\abhij\OneDrive\Pictures\Screenshots\Screenshot (34).png")
coord = detected_face(img2)
for (x,y,w,h) in coord:
    cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),4)


# In[26]:


plt.imshow(img2)
plt.show()


# In[27]:


cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read(0)
    coord = detected_face(frame)
    
    for (x,y,w,h) in coord:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
    
    cv2.imshow("MyFrame",frame)
    k = cv2.waitKey(2)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




