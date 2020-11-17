#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets,svm,metrics,model_selection
from skimage import feature as ft


path = os.path.abspath(os.path.dirname(sys.argv[0]))+'/trash'
categorys = os.listdir(path)
X = []
Y_label= []
for category in categorys:
    images = os.listdir(path+'/'+category)
    for image in images:
        im = ft.hog(Image.open(path+'/'+category+'/'+image).convert('L').crop((64,128,320,384)),
                    orientations=9, 
                    pixels_per_cell=(16,16), 
                    cells_per_block=(2,2), 
                    block_norm = 'L2-Hys', 
                    transform_sqrt = True, 
                    feature_vector=True, 
                    visualize=False
                    )
        X.append(im)
        Y_label.append(category)        
X = np.array(X)
print(X.shape)
Y_label = np.array(Y_label)
Y = LabelEncoder().fit_transform(Y_label)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)

print(y_test.shape)


# In[9]:


C = 0.05
classifier = svm.SVC(kernel = 'linear',C =C)
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)
print("当C=",C,"时，训练集的准确率为：",classifier.score(x_train,y_train),"测试集的准确率为：",classifier.score(x_test,y_test))


# In[2]:


train_data = x_train
train_label = y_train
test_data = x_test
test_label = y_test

index_one = np.where(train_label==0)
data_one = train_data[index_one]
print(data_one.shape)

index_two = np.where(train_label==1)
data_two = train_data[index_two]
print(data_two.shape)

index_three = np.where(train_label==2)
data_three = train_data[index_three]
print(data_three.shape)


# In[37]:


#第一个ovr
x_t = np.vsplit(data_one[:336],3)
x_f = np.append(data_two,data_three,axis = 0)
np.random.shuffle(x_f)
x_f = np.vsplit(x_f[:693],9)
classifier = svm.SVC(kernel = 'linear',C = 1.4)
score_1 = np.zeros(9*test_label.shape[0]).reshape(3,3,test_label.shape[0])
for i in range(3):
    for j in range(3):
        pos = x_t[i]
        neg = x_f[j]
        x = np.append(pos,neg,axis = 0)
        y = np.append(np.ones(pos.shape[0]),np.ones(neg.shape[0])*(-1),axis = 0)
        y_label = np.ones(test_label.shape)*(-1)
        y_label[np.where(test_label==0)]=1
        classifier.fit(x,y)
        print(classifier.score(test_data,y_label))
        score_1[i,j]=classifier.predict(test_data)
#min_max步骤
output_1 = np.max(np.min(score_1,axis=0),axis = 0)
acc= np.where(output_1==y_label)[0].shape[0]/y_label.shape[0]
a=np.where(output_1==1)
print(a[0].shape,acc)


# In[80]:


#22222222
x_t =np.vsplit(data_two[:327],3)
x_f = np.append(data_one,data_three,axis = 0)
np.random.shuffle(x_f)
x_f = np.vsplit(x_f[:702],9)
#模型训练
classifier = svm.SVC(kernel = 'linear',C = 1.7)
score_2 = np.zeros(9*test_label.shape[0]).reshape(3,3,test_label.shape[0])
for i in range(3):
    for j in range(3):
        pos = x_t[i]
        neg = x_f[j]
        x = np.append(pos,neg,axis = 0)
        y = np.append(np.ones(pos.shape[0]),np.ones(neg.shape[0])*(-1),axis = 0)
        y_label = np.ones(test_label.shape)*(-1)
        y_label[np.where(test_label==1)]=1
        classifier.fit(x,y)
        score_2[i,j]=classifier.predict(test_data)
        print(classifier.score(test_data,y_label))
print(np.shape(score_2))
#min_max步骤
output_2= np.max(np.min(score_2,axis=0),axis = 0)

acc= np.where(output_2==y_label)[0].shape[0]/y_label.shape[0]
a=np.where(output_2==1)
print(a[0].shape,acc)


# In[104]:


#333333333333
x_t =np.vsplit(data_three[:369],3)
x_f = np.append(data_one,data_two,axis = 0)
np.random.shuffle(x_f)
x_f = np.vsplit(x_f[:657],9)
classifier = svm.SVC(kernel = 'linear',C =2.6)
score_3 = np.zeros(9*test_label.shape[0]).reshape(3,3,test_label.shape[0])
for i in range(3):
    for j in range(3):
        pos = x_t[i]
        neg = x_f[j]
        x = np.append(pos,neg,axis = 0)
        y = np.append(np.ones(pos.shape[0]),np.ones(neg.shape[0])*(-1),axis = 0)
        y_label = np.ones(test_label.shape)*(-1)
        y_label[np.where(test_label==2)]=1
        classifier.fit(x,y)
        score_3[i,j]=classifier.predict(test_data)
        print(classifier.score(test_data,y_label))
#min_max步骤
output_3= np.max(np.min(score_3,axis=0),axis = 0)


acc= np.where(output_3==y_label)[0].shape[0]/y_label.shape[0]
a=np.where(output_3==1)
print(a[0].shape,acc)


# In[105]:


res = np.zeros(test_label.shape[0]*3).reshape(test_label.shape[0],3)
res[:,0]=output_1
res[:,1]=output_2
res[:,2]=output_3

s = np.sum(res,axis = 1)
out=np.argmax(res,axis=1)

output = np.ones(test_label.shape[0])

output[np.where(output_1==1)] = 0
output[np.where(output_2==1)] = 1
output[np.where(output_3==1)] = 2

print(out.shape)
acc= np.where(output==test_label)[0].shape[0]/test_label.shape[0]
print("准确率为",acc)

