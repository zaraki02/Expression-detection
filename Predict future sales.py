#!/usr/bin/env python
# coding: utf-8

# In[286]:


import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.preprocessing import Imputer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[189]:


train = pandas.read_csv(r"C:\Users\abhij\.anaconda\navigator\Predict future sales\sales_train.csv")
test = pandas.read_csv(r"C:\Users\abhij\.anaconda\navigator\Predict future sales\test.csv")


# In[190]:


train.head


# In[191]:


test.head


# In[192]:


train.isnull().sum()


# In[193]:


test.isnull().sum()


# In[ ]:


# No missing values....


# In[194]:


train["date"] = pandas.to_datetime(train["date"])


# In[197]:


#Extracting the months....
month= []
year= []
day= []
for x in train["date"]:
    month.append(x.strftime("%B"))
    year.append(x.strftime("%y"))
    day.append(x.strftime("%d"))


# In[198]:


#Mapping month to month no...
mon= {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
train["month"] = month
train["month"] = train["month"].map(mon)


# In[199]:


#Extracting the year nos...
train["year"] = year
train["day"] = day


# In[200]:


train.head()


# In[201]:


train = train.drop(["date","date_block_num"],axis=1)
train.head()


# In[202]:


train["count"] = numpy.uint8(train["item_cnt_day"])
train =train.drop(["item_cnt_day"],axis=1)


# In[203]:


train.head()


# In[210]:


test = test.drop(["ID"],axis=1)
test.head()


# In[211]:


test.shape


# In[212]:


train.shape


# In[213]:


train["year"].hist()


# In[214]:


train["month"].hist()


# In[215]:


train["day"].hist()


# In[216]:


data = pandas.concat([train,test],axis=0)
data.head()


# In[217]:


data.isnull().sum()
numpy.array(data["count"]).shape
data["count"] = numpy.uint8(data["count"])
data.head()


# In[218]:


imp_mean = Imputer(missing_values="NaN" ,strategy="mean",axis=0)
cnt = numpy.reshape(numpy.array(data["count"]),((numpy.array(data["count"]).shape[0]),1))
#nt = numpy.array(data["count"]).reshape(numpy.array(data["count"]).shape[0]
cnt


# In[219]:


data["count"] = imp_mean.fit_transform(cnt)
data["count"]


# In[220]:


data.isnull().sum()


# In[221]:


imp_mode = Imputer(missing_values = "NaN", strategy="most_frequent", axis=0)
day=numpy.reshape(numpy.array(data["day"]),(numpy.array(data["day"]).shape[0],1))
month= numpy.reshape(numpy.array(data["month"]),(numpy.array(data["month"]).shape[0],1))
year= numpy.reshape(numpy.array(data["year"]),(numpy.array(data["year"]).shape[0],1))


# In[222]:


data["day"] =imp_mode.fit_transform(day)
data["month"] =imp_mode.fit_transform(month)
data["year"] =imp_mode.fit_transform(year)


# In[223]:


data.isnull().sum()


# In[224]:


price = numpy.reshape(numpy.array(data["item_price"]),(numpy.array(data["item_price"]).shape[0],1))
data["item_price"] = imp_mean.fit_transform(price)


# In[225]:


data.isnull().sum()


# In[ ]:


#No more null values....


# In[137]:


corr = data.corr()


# In[138]:


plt.matshow(corr)


# In[226]:


data.head()


# In[240]:


data = data.drop(["day"],axis=1)


# In[241]:


from sklearn.preprocessing import LabelEncoder


# In[247]:


label_enc = LabelEncoder()
data["month"] =label_enc.fit_transform(data["month"])
data["year"] = label_enc.fit_transform(data["year"])


# In[248]:


data.head(10)


# In[249]:


data["item_id"].value_counts()


# In[252]:


label_enc = LabelEncoder()
data["item_id"] = label_enc.fit_transform(data["item_id"])
data["shop_id"] = label_enc.fit_transform(data["shop_id"])


# In[253]:


data.head(10)


# In[257]:


len(data["item_id"].unique())


# In[258]:


max(data["item_id"])


# In[256]:


len(data["item_id"])


# In[260]:


data["sale"] =data["count"]
data =data.drop(["count"],axis=1)
data.head(10)


# In[278]:


data["sale"] = numpy.uint8(data["sale"])
train_X = data.iloc[:train.shape[0],:5]
train_Y = data.iloc[:train.shape[0],5]

test_X = data.iloc[test.shape[0]+1:,:5]
test_Y = data.iloc[test.shape[0]+1:,5]


# In[279]:


train_X.head()


# In[280]:


train_Y.head()


# In[281]:


model = LinearRegression()


# In[282]:


model.fit(train_X,train_Y)


# In[283]:


test_X.head()


# In[284]:


y = model.predict(test_X)


# In[285]:


y


# In[287]:


r2_score(y,test_Y)


# In[288]:


r2_score(test_Y,y)


# In[293]:


model.score(test_X,test_Y)


# In[ ]:




