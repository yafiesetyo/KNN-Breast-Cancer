#%%
#import csv

import pandas as pd

dataset = pd.read_csv('breast-cancer(3).csv')
dataset

#%%
#data preprocessing #1 (search for some unknown data, then drop the rows that have it (unknown data))

dt_clean = dataset.drop((dataset.loc[dataset['node-caps']=='?']).index)
dt_clean

# %%
#data preprocessing #2 (encode str data into int datatype, because when i used sklearn, they didnt approved string data)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

c = le.fit_transform(dt_clean['class'])
age = le.fit_transform(dt_clean['age'])
menopause = le.fit_transform(dt_clean['menopause'])
tumor_size = le.fit_transform(dt_clean['tumor-size'])
inv_nodes = le.fit_transform(dt_clean['inv-nodes'])
node_caps = le.fit_transform(dt_clean['node-caps'])
deg_malig = le.fit_transform(dt_clean['deg-malig'])
breast = le.fit_transform(dt_clean['breast'])
breast_quad = le.fit_transform(dt_clean['breast-quad'])
irradiat = le.fit_transform(dt_clean['irradiat'])

a = zip(age,menopause,tumor_size,inv_nodes,node_caps,deg_malig,breast,breast_quad,irradiat)

a

#%%
# data preprocessing #3 (splitting data into data test and data train)
from sklearn.model_selection import train_test_split as tt 

cl = c.tolist()
att = list(a)
att_train,att_test,cl_train,cl_test = tt(att,cl,test_size=0.2,random_state=0)
att_train


# %%
#knn started...
#first we search optimal k number that soon we will used
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import f1_score as f1
import matplotlib.pyplot as pl

skor = []
k_range = range(1,64)
for i in k_range:
    mod = knn(n_neighbors=i)
    mod.fit(att_train,cl_train)
    pred = mod.predict(att_test)
    skor.append(f1(cl_test,pred,average='macro'))

pl.plot(k_range,skor)
pl.xlabel('nilai k')
pl.ylabel('F1 Macro Score')
pl.show()


# %%
#then we got k=63-64 (between them hehe). The Real KNN starts here...

model = knn(n_neighbors=64)
model.fit(att_train,cl_train)
predict = model.predict(att_test)
predict

#%%
#count accuracy and f1 score
from sklearn.metrics import accuracy_score as acc

akurasi = acc(cl_test,predict)
f1_skor = f1(cl_test,pred,average='macro')
print ("Accuracy : ",akurasi)
print ("F1-Macro Score : ",f1_skor)

# %%
#lets test with brain-randomed data hehe

hooman = [[4,1,1,1,2,1,3,1,0],[3,2,1,1,1,1,1,1,1]]
answer = model.predict(hooman)
answer

# %%
