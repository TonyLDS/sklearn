# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 03:56:07 2017

@author: luzhangqin
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components = 2)
reduced_X = pca.fit_transform(X)

red_X, red_y = [], []
blue_X, blue_y = [], []
green_X, green_y = [], []

for i in range(len(reduced_X)):
    if y[i]  == 0:
        red_X.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i]  == 1:
        blue_X.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_X.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
        
plt.scatter(red_X, red_y, c = 'r', marker = 'x')
plt.scatter(blue_X, blue_y, c = 'b', marker = 'D')
plt.scatter(green_X, green_y, c = 'g', marker = '.')
plt.show()
