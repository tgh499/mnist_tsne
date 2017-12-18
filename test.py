import numpy as np


dim = 28
X_max = 18.6436675649
Y_max = 18.4328531037
X_min = 0.00318510345977
Y_min = 0.00318510345662

Z1 = 1
Z2 = 1


#oneD_mapping = []

temp = []

temp0 = (Z1 - X_min) / ((X_max - X_min) / (dim-1))
temp1 = (Z2 - Y_min) / ((Y_max - Y_min) / (dim-1))



print("CORRECT")
print(np.round(temp0))
print(np.round(temp1))

temp = np.round(temp1) * dim + np.round(temp0)
print(int(temp))


print("WRONG")
print(temp0)
print(temp1)
temp = int(np.round(temp1 * dim + temp0))
print(temp)

#oneD_mapping.append(int(np.round(temp)))
