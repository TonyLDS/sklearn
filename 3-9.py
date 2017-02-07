# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:43:47 2017

@author: luzhangqin
"""

from sklearn import preprocessing
import numpy as np
X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.],
])
#标准化
print(preprocessing.scale(X))