#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from sklearn import svm
import os
import sys

train_data = np.load(os.path.abspath(os.path.dirname(sys.argv[0]))+"/train_data.npy")
train_label = np.load(os.path.abspath(os.path.dirname(sys.argv[0]))+"/train_label.npy")
test_data = np.load(os.path.abspath(os.path.dirname(sys.argv[0]))+"/test_data.npy")
test_label = np.load(os.path.abspath(os.path.dirname(sys.argv[0]))+"/test_label.npy")
print(train_data.shape,test_data.shape)
C = 1
classifier = svm.SVC(kernel = 'linear',C = C,random_state = 0).fit(train_data,train_label)
print("训练精度：",classifier.score(train_data,train_label),"测试精度",classifier.score(test_data,test_label))


# In[2]:


#random decomposition
#划分数据
#标签分开
index_one = np.where(train_label==0)
data_one = train_data[index_one]

index_two = np.where(train_label==1)
data_two = train_data[index_two]

index_three = np.where(train_label==-1)
data_three = train_data[index_three]

#one_vs_rest_1
np.random.shuffle(data_one)
x_t =np.vsplit(data_one,4)
x_f = np.append(data_two,data_three,axis = 0)
np.random.shuffle(x_f)
x_f = np.vsplit(x_f[:25216],8)
cclassifier = svm.SVC(kernel = 'linear',C = 0.005,random_state = 0)
score_1 = np.zeros(16*test_label.shape[0]).reshape(4,4,test_label.shape[0])
for i in range(4):
    for j in range(4):
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


# In[3]:


#one_vs_rest_2
#数据处理
np.random.shuffle(data_two)
x_t =np.vsplit(data_two[:12896],4)
x_f = np.append(data_one,data_three,axis = 0)
np.random.shuffle(x_f)
x_f = np.vsplit(x_f[:24464],8)
#模型训练
classifier = svm.SVC(kernel = 'linear',C = 0.05,random_state = 0)
score_2 = np.zeros(16*test_label.shape[0]).reshape(4,4,test_label.shape[0])
for i in range(4):
    for j in range(4):
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


# In[8]:


#one_vs_rest_3
#数据处理
np.random.shuffle(data_three)
x_t =np.vsplit(data_three,4)
x_f = np.append(data_one,data_two,axis = 0)
np.random.shuffle(x_f)
x_f = np.vsplit(x_f[:25040],8)
#模型训练
classifier = svm.SVC(kernel = 'linear',C =0.0001,random_state = 0)
score_3 = np.zeros(16*test_label.shape[0]).reshape(4,4,test_label.shape[0])
for i in range(4):
    for j in range(4):
        pos = x_t[i]
        neg = x_f[j]
        x = np.append(pos,neg,axis = 0)
        y = np.append(np.ones(pos.shape[0]),np.ones(neg.shape[0])*(-1),axis = 0)
        classifier.fit(x,y)
        score_3[i,j]=classifier.predict(test_data)
        print(classifier.score(x,y))
#min_max步骤
output_3= np.max(np.min(score_3,axis=0),axis = 0)
acc= np.where(output_3==y_label)[0].shape[0]/y_label.shape[0]
a=np.where(output_3==1)
print(a[0].shape,acc)


# In[5]:


res = np.zeros(test_label.shape[0]*3).reshape(test_label.shape[0],3)
res[:,0]=output_3
res[:,1]=output_1
res[:,2]=output_2

s = np.sum(res,axis = 1)
out=np.argmax(res,axis=1)-1

classifier_2 = svm.SVC(kernel = 'linear',C =0.0001,random_state = 0).fit(train_data,train_label)
unsure = classifier_2.predict(test_data[np.where(s==-3)])

out[np.where(s==-3)]=unsure
print(out.shape)
acc= np.where(out==test_label)[0].shape[0]/test_label.shape[0]
print(acc)


# In[1]:


# task decomposition with prior knowledge strategies. 
#划分数据
#标签分开
index_one = np.where(train_label==0)
data_one = train_data[index_one]

index_two = np.where(train_label==1)
data_two = train_data[index_two]

index_three = np.where(train_label==-1)
data_three = train_data[index_three]

#one_vs_rest_1
np.random.shuffle(data_one)
x_t =np.vsplit(data_one,4)
x_f = np.append(data_two,data_three,axis = 0)
np.random.shuffle(x_f)
#先验知识，求第一维期望，选出期望附近的balance——number
mean_one = np.mean(x_f[:,0])
i = np.where(x_f[:,0]>mean_one-0.85)
j = np.where(x_f[:,0]<mean_one+0.85)
inter = np.intersect1d(i,j)
x_f = x_f[inter]
print(x_f.shape)
x_f = np.vsplit(x_f[:13160],4)
classifier = svm.SVC(kernel = 'linear',C = 0.005,random_state = 0)
score_1 = np.zeros(16*test_label.shape[0]).reshape(4,4,test_label.shape[0])
for i in range(4):
    for j in range(4):
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


# In[4]:


#one_vs_rest_2
#数据处理
np.random.shuffle(data_two)
x_t =np.vsplit(data_two[:12896],4)
x_f = np.append(data_one,data_three,axis = 0)
np.random.shuffle(x_f)
#先验知识，求第一维期望，选出期望附近的balance——number
mean_one = np.mean(x_f[:,0])
i = np.where(x_f[:,0]>mean_one-0.59)
j = np.where(x_f[:,0]<mean_one+0.59)
inter = np.intersect1d(i,j)
x_f = x_f[inter]
print(x_f.shape)
x_f = np.vsplit(x_f[:12516],4)
#模型训练
classifier = svm.SVC(kernel = 'linear',C = 0.05,random_state = 0)
score_2 = np.zeros(16*test_label.shape[0]).reshape(4,4,test_label.shape[0])
for i in range(4):
    for j in range(4):
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


# In[66]:


#one_vs_rest_3
#数据处理
np.random.shuffle(data_three)
x_t =np.vsplit(data_three,4)
x_f = np.append(data_one,data_two,axis = 0)
np.random.shuffle(x_f)
mean_one = np.mean(x_f[:,0])
i = np.where(x_f[:,0]>mean_one-0.77)
j = np.where(x_f[:,0]<mean_one+0.77)
inter = np.intersect1d(i,j)
x_f = x_f[inter]
print(x_f.shape)
x_f = np.vsplit(x_f[:12400],4)
#模型训练
classifier = svm.SVC(kernel = 'linear',C = 0.0001,random_state = 0)
score_3 = np.zeros(16*test_label.shape[0]).reshape(4,4,test_label.shape[0])
for i in range(4):
    for j in range(4):
        pos = x_t[i]
        neg = x_f[j]
        x = np.append(pos,neg,axis = 0)
        y = np.append(np.ones(pos.shape[0]),np.ones(neg.shape[0])*(-1),axis = 0)
        y_label = np.ones(test_label.shape)*(-1)
        y_label[np.where(test_label==-1)]=1
        classifier.fit(x,y)
        score_3[i,j]=classifier.predict(test_data)
        print(classifier.score(test_data,y_label))
#min_max步骤
output_3= np.max(np.min(score_3,axis=0),axis = 0)
acc= np.where(output_3==y_label)[0].shape[0]/y_label.shape[0]
a=np.where(output_3==1)
print(a[0].shape,acc)


# In[65]:


res = np.zeros(test_label.shape[0]*3).reshape(test_label.shape[0],3)
res[:,0]=output_3
res[:,1]=output_1
res[:,2]=output_2

s = np.sum(res,axis = 1)
out=np.argmax(res,axis=1)-1

classifier_2 = svm.SVC(kernel = 'linear',C =0.0001,random_state = 0).fit(train_data,train_label)
unsure = classifier_2.predict(test_data[np.where(s==-3)])

out[np.where(s==-3)]=unsure
print(out.shape)
acc= np.where(out==test_label)[0].shape[0]/test_label.shape[0]
print(acc)

