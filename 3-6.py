# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:17:07 2017

@author: luzhangqin
"""

from sklearn import datasets

import matplotlib.pyplot as plt

digits = datasets.load_digits()
print('Digits: ', digits.target[0])
print(digits.images[0])
print('Feature vector:\n', digits.images[0].reshape(-1, 64))
plt.figure()
plt.axis('off')
#plt.cm.gray_r灰度图
#interpolation
plt.imshow(digits.images[0], cmap = plt.cm.gray_r, interpolation = 'nearest')
plt.show()