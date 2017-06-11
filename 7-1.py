# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 01:47:09 2017

@author: luzhangqin
"""

import numpy as np
X = [[2, 0, -1.4],
     [2.2, 0.2, -1.5],
     [2.4, 0.1, -1],
     [1.9, 0, -1.2]]
print(np.cov(np.array(X).T))
print('-------------------------------------')
w, v = np.linalg.eig(np.array([[1, -2], [2, -3]]))
print('特征值：', w)
print('特征向量：', v)
print('-------------------------------------')
arr = np.array([[0.9, 2.4, 1.2, 0.5, 0.3, 1.8, 0.5, 0.3, 2.5, 1.3],
              [1.0, 2.6, 1.7, 0.7, 0.7, 1.4, 0.6, 0.6, 2.6, 1.1]])
cov = np.cov(arr)
print('协方差矩阵：', cov)
w, v = np.linalg.eig(cov)
print('特征值：', w)
print('特征向量：', v)
v = v.reshape((-1,1))[::2]
arr[0] -= np.mean(arr[0])
arr[1] -= np.mean(arr[1])
arr = arr.reshape((2,-1)).T
print('arr:', arr)
print('v:', v)
pca = np.dot(arr, v)
print('PCA:', pca)