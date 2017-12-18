import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import random

mapping = pd.read_csv('mappedX_KL.csv', header=None)
mapping = mapping.as_matrix()

test = pd.read_csv('mnist_5000.csv', header=None)
features_train = test.columns[1:]
label_train = test.columns[0]
X_train = test[features_train]
y_train = test[label_train]
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

for i,j in enumerate(mapping):
    mapping[i][0] = j[0] + 2.448865081859
mapping = mapping + 6.34

'''
for i,j in enumerate(mapping):
    mapping[i][0] = np.round(j[0])
    mapping[i][1] = np.round(j[1])
'''

X = []
Y = []
oneD_mapping = []
dim = 28
count = 0
X_max = 18.6436675649
Y_max = 18.4328531037
X_min = 0.00318510345977
Y_min = 0.00318510345662
count = 0
for i in mapping:
    temp = []
    temp0 = (i[0] - X_min) / ((X_max - X_min) / (dim-1))
    temp1 = (i[1] - Y_min) / ((Y_max - Y_min) / (dim-1))
    ############ RED FLAG
    temp = np.round(temp1) * dim + np.round(temp0)
    #temp = temp1 * dim + temp0
    if count == 128:
        print(temp0)
        print(temp1)
        print(temp)
        print(np.round(temp))
        print(int(np.round(temp)))
    count += 1
    oneD_mapping.append(int(temp))
    #oneD_mapping.append(int(np.round(temp)))

tsne_mapped_Xtrain = np.zeros((len(X_train), (len(X_train[0]))))
############ RED FLAG

#tsne_mapped_Xtrain = np.zeros(len(X_train[0]))


#for i,j in enumerate(oneD_mapping):
    #tsne_mapped_Xtrain[j] = xc[i]

for i,j in enumerate(tsne_mapped_Xtrain):
    for k,l in enumerate(oneD_mapping):
        tsne_mapped_Xtrain[i][l] = X_train[i][k]


label_fm = pd.DataFrame(y_train)
label_fm.to_csv('lt_train.csv', encoding='utf-8', index=False)

data_fm = pd.DataFrame(tsne_mapped_Xtrain)
data_fm.to_csv('dt_train.csv', encoding='utf-8', index=False)


# to plot an image corresponding to sample_index
sample_index = 12
C = []
X = []
Y = []
for i in mapping:
    X.append(i[0])
    Y.append(i[1])
for i in X_train[sample_index]:
    C.append(i)

plt.scatter(X, Y, c = C, cmap='gray_r')
plt.show()



#arr = np.reshape(X_train[sample_index], (dim,dim))
arr = np.reshape(tsne_mapped_Xtrain[sample_index], (dim,dim))
#arr = np.reshape(tsne_mapped_Xtrain, (dim,dim))
plt.imshow(arr, cmap='gray_r')
plt.gca().invert_yaxis()
#plt.colorbar()
plt.show()
