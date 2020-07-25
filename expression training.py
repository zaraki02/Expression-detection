#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau


# In[2]:


Train = pd.read_csv(r"C:\Users\abhij\Facial Expressions\train.csv")
Test = pd.read_csv(r"C:\Users\abhij\Facial Expressions\test.csv")


# In[3]:


print(Train.shape)
train =Train[:]# taking a small part of Train and Test
test = Test[:]
print(train["emotion"].unique())
print(train.shape)
print(test.shape)
train.shape
train.head(10)


# In[4]:


#splitting the data...
x_train,x_val,y_train,y_val = train_test_split(train["pixels"],train["emotion"],test_size=0.2)


# In[5]:


#Got all the pixel values in a list
train_pixels = []
test_pixels =[]
val_pixels = []
for x in x_train:
    train_pixels.append(x.split(" "))

for x in x_val:
    val_pixels.append(x.split(" "))

#For prediction...
for x in test.pixels:
    test_pixels.append(x.split(" "))
#print(train_pixels[0])
#pint(val_pixels[0])
#pint(test_pixels[0])


# In[6]:


print(len(train_pixels[0]))
print(numpy.array(train_pixels).shape)


# In[7]:


train_pixels = numpy.array(train_pixels,"float32")
train_pixels = train_pixels.reshape(train_pixels.shape[0],48,48,1)#last 1 for color dimension(grey-1)
print(train_pixels.shape)

val_pixels = numpy.array(val_pixels,"float32")
val_pixels = val_pixels.reshape(val_pixels.shape[0],48,48,1)#last 1 for color dimension(grey-1)
print(val_pixels.shape)

test_pixels = numpy.array(test_pixels,"float32")
test_pixels = test_pixels.reshape(test_pixels.shape[0],48,48,1)#last 1 for color dimension(grey-1)
print(test_pixels.shape)


# In[8]:


#plt.imshow(train_pixels[0])
#print(train_pixels[0])
#print(train_pixels.dtype)
#print(type(train_pixels))\

#One hot encoding for the emotions....
emotions_train = to_categorical(y_train)
emotions_val = to_categorical(y_val)
print(emotions_train.shape,"   ",emotions_train[0])
print(emotions_val.shape,"   ",emotions_val[0])


# In[9]:


#normalizing the data...

minm = numpy.min(train_pixels)
maxm = numpy.max(train_pixels)

num = train_pixels-minm
train_x = num/(maxm-minm)

#normalizing the data...
minm = numpy.min(val_pixels)
maxm = numpy.max(val_pixels)

num = val_pixels-minm
val_x = num/(maxm-minm)

#normalizing the data...
minm = numpy.min(test_pixels)
maxm = numpy.max(test_pixels)

num = test_pixels-minm
test_x = num/(maxm-minm)

#print(train_pixels_x[:5])
#print(numpy.min(train_pixels_x),minm)


# In[10]:


#Creating the CNN for training...
#padding is same and keep it so...
model = Sequential()
model.add(Conv2D(32,(3,3),strides=(1, 1),padding='same'))
model.add(Conv2D(64,(3,3),strides=(1, 1),padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3),strides=(1, 1),padding='same'))
model.add(Conv2D(64,(3,3),strides=(1, 1),padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128,(3,3),strides=(1, 1),padding='same'))
model.add(Conv2D(128,(3,3),strides=(1, 1),padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256,(3,3),strides=(1, 1),padding='same'))
model.add(Conv2D(512,(3,3),strides=(1, 1),padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

#Dense layer...
model.add(Dense(256,activation ="relu",input_shape=(48,48,1)))
model.add(Dropout(0.4))
model.add(Dense(512,activation ="relu"))
model.add(Dropout(0.4))
model.add(Dense(7,activation ="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[11]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)
history = model.fit(train_x,emotions_train,batch_size=64,callbacks=[reduce_lr],epochs=50,verbose=1,validation_split=0.2, validation_data=(val_x,emotions_val), shuffle=True)


# In[12]:


#Creating the CNN for training...
#padding is same and keep it so...
model2 = Sequential()
model2.add(Conv2D(32,(3,3),strides=(1, 1),padding='same'))
model2.add(Conv2D(64,(3,3),strides=(1, 1),padding='same'))
model2.add(MaxPool2D(pool_size=(2, 2)))

model2.add(Conv2D(64,(3,3),strides=(1, 1),padding='same'))
model2.add(Conv2D(64,(3,3),strides=(1, 1),padding='same'))
model2.add(MaxPool2D(pool_size=(2, 2)))

model2.add(Conv2D(128,(3,3),strides=(1, 1),padding='same'))
model2.add(Conv2D(128,(3,3),strides=(1, 1),padding='same'))
model2.add(MaxPool2D(pool_size=(2, 2)))

model2.add(Conv2D(256,(3,3),strides=(1, 1),padding='same'))
model2.add(Conv2D(512,(3,3),strides=(1, 1),padding='same'))
model2.add(MaxPool2D(pool_size=(2, 2)))

model2.add(Flatten())

#Dense layer...
model2.add(Dense(256,activation ="relu",input_shape=(48,48,1)))
model2.add(Dropout(0.4))
model2.add(Dense(512,activation ="relu"))
model2.add(Dropout(0.4))
model2.add(Dense(7,activation ="softmax"))

model2.compile(optimizer="SGD",loss="categorical_crossentropy",metrics=["accuracy"])


# In[13]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
#model.summary()
history2 = model2.fit(train_x,emotions_train,batch_size=64,callbacks=[reduce_lr],epochs=50,verbose=1,validation_split=0.2, validation_data=(val_x,emotions_val), shuffle=True)


# In[1]:


results = model.predict_classes(val_pixels)
print(results[:50])
print(y_val)
#print(model.evaluate(val_pixels,results,batch_size=32))


# In[15]:


model2.save("exp.h5")


# In[3]:


from keras.models import load_model


# In[4]:


model = load_model("exp.h5")


# In[5]:


model.summary()


# In[6]:


res =model.predict(val_pixels)
res_=model.predict_classes(val_pixels)
print(res_)


# In[20]:


res.shape


# In[ ]:




